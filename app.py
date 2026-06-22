import streamlit as st
import os
import json
import io
import csv
from datetime import datetime, timedelta
from st_copy_to_clipboard import st_copy_to_clipboard
from utils import RSS_FEEDS, fetch_top_stories, rewrite_with_groq, rewrite_with_gemini, rewrite_with_nemotron

st.set_page_config(page_title="Newsdrum AI Aggregator Panel", layout="wide")

LOCK_FILE = ".csv_lock.json"

# ── 24-hour lock helpers ──────────────────────────────────────
def can_export():
    """Returns (allowed: bool, time_remaining: str | None)"""
    try:
        if os.path.exists(LOCK_FILE):
            with open(LOCK_FILE, "r") as f:
                data = json.load(f)
            last = datetime.fromisoformat(data["last_exported"])
            diff = datetime.now() - last
            if diff < timedelta(hours=24):
                remaining = timedelta(hours=24) - diff
                h = int(remaining.total_seconds() // 3600)
                m = int((remaining.total_seconds() % 3600) // 60)
                return False, f"{h}h {m}m"
    except Exception:
        pass
    return True, None

def mark_exported():
    with open(LOCK_FILE, "w") as f:
        json.dump({"last_exported": datetime.now().isoformat()}, f)

def build_csv(rows):
    """rows = list of [source_news, ai_headline, extension]"""
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Source News", "News Headline", "Extension"])
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")

# ── Session state init ────────────────────────────────────────
if "active_source" not in st.session_state:
    st.session_state.active_source = None
if "csv_rows" not in st.session_state:
    st.session_state.csv_rows = []

# ── CSS ───────────────────────────────────────────────────────
st.markdown("""
    <style>
    .stApp { background-color: #f8fafc; }
    h1, h2, h3 { color: #0f172a; font-family: 'Inter', sans-serif; }

    [data-testid="stExpander"] {
        background-color: #ffffff;
        border: 1px solid #cbd5e1;
        border-radius: 8px;
    }
    [data-testid="stExpander"] details summary p {
        color: #0f172a !important;
        font-weight: 700 !important;
    }
    [data-testid="stExpander"] div { color: #334155 !important; }

    .model-badge {
        font-size: 12px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 10px;
        color: #64748b;
    }
    .meta-headline {
        font-size: 18px;
        font-weight: 800;
        color: #0f172a;
        line-height: 1.3;
        margin-bottom: 12px;
    }
    .meta-strapline {
        font-size: 14px;
        color: #ea580c;
        font-weight: 600;
        line-height: 1.4;
        border-left: 3px solid #ea580c;
        padding-left: 10px;
        margin-bottom: 12px;
    }
    .news-body {
        font-size: 14px;
        color: #334155;
        line-height: 1.6;
        word-wrap: break-word;
    }
    .source-banner {
        background: #e2e8f0;
        padding: 10px 15px;
        border-radius: 6px;
        font-size: 14px;
        font-weight: 600;
        color: #334155;
        margin-bottom: 15px;
    }

    /* Orange Export CSV button */
    [data-testid="stDownloadButton"] > button {
        background-color: #ea580c !important;
        color: #ffffff !important;
        border: none !important;
        font-weight: 700 !important;
        width: 100% !important;
        border-radius: 6px !important;
    }
    [data-testid="stDownloadButton"] > button:hover {
        background-color: #c2410c !important;
        color: #ffffff !important;
    }

    /* Disabled export button */
    .export-disabled button {
        background-color: #94a3b8 !important;
        color: #ffffff !important;
        border: none !important;
        font-weight: 600 !important;
        width: 100% !important;
        border-radius: 6px !important;
        cursor: not-allowed !important;
    }
    </style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.title("📰 Newsdrum")
    st.caption("AI Aggregator Panel v2.1")
    st.write("---")
    refresh_clicked = st.button(
        "🔄 Fetch Latest News",
        use_container_width=True,
        type="primary"
    )
    st.write("### TRACKED SOURCES")
    selected_source = st.radio(
        "Select News Portal",
        options=list(RSS_FEEDS.keys()),
        label_visibility="collapsed"
    )

# ── Reset CSV rows when source changes or refresh clicked ─────
if selected_source != st.session_state.active_source or refresh_clicked:
    st.session_state.active_source = selected_source
    st.session_state.csv_rows = []

# ── Header row: Live Feed title + Export button ───────────────
header_col, btn_col = st.columns([5, 1])

with header_col:
    st.subheader(f"⚡ Live Feed: {selected_source}")

with btn_col:
    allowed, time_left = can_export()
    has_data = len(st.session_state.csv_rows) > 0

    if allowed and has_data:
        fname = (
            f"newsdrum_{selected_source.replace(' ', '_')}_"
            f"{datetime.now().strftime('%Y%m%d')}.csv"
        )
        csv_bytes = build_csv(st.session_state.csv_rows)
        downloaded = st.download_button(
            label="📥 Export CSV",
            data=csv_bytes,
            file_name=fname,
            mime="text/csv",
            use_container_width=True,
        )
        if downloaded:
            mark_exported()

    elif not allowed:
        st.markdown('<div class="export-disabled">', unsafe_allow_html=True)
        st.button(
            f"⏳ {time_left}",
            disabled=True,
            use_container_width=True,
            help="You can only export one CSV per 24 hours."
        )
        st.markdown('</div>', unsafe_allow_html=True)

    else:
        # News not loaded yet
        st.markdown('<div class="export-disabled">', unsafe_allow_html=True)
        st.button(
            "📥 Export CSV",
            disabled=True,
            use_container_width=True,
            help="Fetch news first to enable export."
        )
        st.markdown('</div>', unsafe_allow_html=True)

# ── Main content ──────────────────────────────────────────────
if selected_source:
    with st.spinner(f"Intercepting top stories from {selected_source}..."):
        raw_items = fetch_top_stories(selected_source, limit=5)

    if not raw_items:
        st.error("Connection blocked by source firewall. Try a different outlet.")
    else:
        for idx, item in enumerate(raw_items):
            st.markdown("---")

            st.markdown(
                f'<div class="source-banner">'
                f'📦 Source: {selected_source} | 🕒 {item["published"]}'
                f'</div>',
                unsafe_allow_html=True
            )

            with st.expander(f"Original Article: {item['title']}"):
                st.markdown(item['description'], unsafe_allow_html=True)
                st.link_button("🔗 View Original Source", item["link"])

            with st.spinner("AI is writing (Token Saver Mode active)..."):
                col1_data = rewrite_with_groq(item["title"], item["description"])
                col2_data = rewrite_with_groq(item["title"], item["description"])
                col3_data = rewrite_with_groq(item["title"], item["description"])

            # Accumulate for CSV (using col1 — Gemini slot)
            # Only add once per item (guard against duplicate reruns)
            already_added = any(
                row[0] == item["title"]
                for row in st.session_state.csv_rows
            )
            if not already_added:
                st.session_state.csv_rows.append([
                    item["title"],          # Source News
                    col1_data["headline"],  # News Headline
                    col1_data["body"],      # Extension / Summary
                ])

            col1, col2, col3 = st.columns(3)

            def render_native_card(model_label, data, column_ref, key_suffix):
                with column_ref:
                    with st.container(border=True):
                        st.markdown(f"""
                            <div class="model-badge">{model_label}</div>
                            <div class="meta-headline">{data['headline']}</div>
                            <div class="meta-strapline">{data['strapline']}</div>
                            <div class="news-body">{data['body']}</div>
                            <br>
                        """, unsafe_allow_html=True)
                        payload = (
                            f"TITLE: {data['headline']}\n"
                            f"STRAPLINE: {data['strapline']}\n\n"
                            f"{data['body']}"
                        )
                        st_copy_to_clipboard(
                            payload,
                            before_copy_label=f"📋 Copy {model_label}",
                            after_copy_label="✅ Copied!",
                            key=f"copy_{idx}_{key_suffix}"
                        )

            render_native_card("Gemini 2.5 Flash",   col1_data, col1, "1")
            render_native_card("Nvidia Nemotron 70B", col2_data, col2, "2")
            render_native_card("Groq Llama 3.1",      col3_data, col3, "3")
