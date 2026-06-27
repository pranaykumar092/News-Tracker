import streamlit as st
import io
import re
from collections import Counter
from datetime import datetime
from openpyxl import Workbook
from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
from openpyxl.utils import get_column_letter
from st_copy_to_clipboard import st_copy_to_clipboard
from utils import RSS_FEEDS, fetch_top_stories, calculate_accuracy_with_groq, rewrite_with_gemini, rewrite_with_nvidia

st.set_page_config(page_title="Newsdrum AI Aggregator Panel", layout="wide")

# ── Session state init ────────────────────────────────────────
if "active_source" not in st.session_state:
    st.session_state.active_source = None
if "csv_rows" not in st.session_state:
    st.session_state.csv_rows = []
if "publish_state" not in st.session_state:
    st.session_state.publish_state = "idle"   # idle | loading | ready
if "publish_excel" not in st.session_state:
    st.session_state.publish_excel = None
if "trending_state" not in st.session_state:
    st.session_state.trending_state = "idle"  # idle | loading | ready
if "trending_results" not in st.session_state:
    st.session_state.trending_results = []
if "live_feed_cache" not in st.session_state:
    st.session_state.live_feed_cache = {}

# ── Excel builder ─────────────────────────────────────────────
def build_excel(rows):
    wb = Workbook()
    ws = wb.active
    ws.title = "Newsdrum Export"

    headers = ["Source", "Source link", "Source News", "NewsDrum Version"]
    col_widths = [len(h) for h in headers]

    def thin():
        s = Side(style="thin", color="CCCCCC")
        return Border(left=s, right=s, top=s, bottom=s)

    for c, h in enumerate(headers, 1):
        cell = ws.cell(row=1, column=c, value=h)
        cell.font      = Font(name="Arial", bold=True, size=11, color="FFFFFF")
        cell.fill      = PatternFill("solid", start_color="1F2D3D")
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.border    = thin()
    ws.row_dimensions[1].height = 22

    for r, row in enumerate(rows, 2):
        max_lines = 1
        for c, val in enumerate(row, 1):
            text = str(val) if val else ""
            
            # Format Source link column as a clickable hyperlink to save space
            if c == 2 and text.startswith("http"):
                cell = ws.cell(row=r, column=c, value="View Source")
                cell.hyperlink = text
                cell.font = Font(name="Arial", size=10, color="0563C1", underline="single")
                display_text = "View Source"
            else:
                cell = ws.cell(row=r, column=c, value=text)
                cell.font = Font(name="Arial", size=10)
                display_text = text

            cell.alignment = Alignment(vertical="top", wrap_text=True)
            cell.border    = thin()
            longest_line   = max((len(line) for line in display_text.split("\n")), default=0)
            col_widths[c-1] = min(80, max(col_widths[c-1], longest_line))
            cap   = col_widths[c-1] if col_widths[c-1] > 0 else 1
            lines = sum(max(1, (len(line)+cap-1)//cap) for line in display_text.split("\n"))
            max_lines = max(max_lines, lines)
        ws.row_dimensions[r].height = min(400, max(18, max_lines * 15))

    for i, w in enumerate(col_widths, 1):
        ws.column_dimensions[get_column_letter(i)].width = w + 4
    ws.freeze_panes = "A2"

    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)
    return buf.getvalue()

def make_payload(data):
    return (
        f"TITLE: {data['headline']}\n"
        f"STRAPLINE: {data['strapline']}\n\n"
        f"{data['body']}"
    )

def is_error(data):
    """True when a rewrite_* result is an API/token/rate-limit error.
    On error the rewrite functions set strapline to e.g. 'Gemini Error'."""
    return str(data.get("strapline", "")).strip().endswith("Error")

# ── Common-story detection ────────────────────────────────────
# Words too generic to help identify a story — ignored when matching.
STOPWORDS = {
    "the","a","an","in","on","at","of","for","to","and","or","but","with",
    "is","are","was","were","be","been","as","by","from","that","this","it",
    "his","her","its","their","they","he","she","we","you","not","no","new",
    "after","before","over","under","into","out","up","down","says","say",
    "said","will","has","have","had","amid","among","more","than","what",
    "who","how","why","when","where","amp","get","gets","may","can","could",
    "would","should","his","s","t","vs","top","day","year","years","news"
}

def _keywords(title):
    """Reduce a headline to its set of meaningful keywords."""
    words = re.findall(r"[a-z0-9]+", title.lower())
    return {w for w in words if w not in STOPWORDS and len(w) > 2}

def _similarity(kw1, kw2):
    """Jaccard similarity between two keyword sets (0.0 – 1.0)."""
    if not kw1 or not kw2:
        return 0.0
    return len(kw1 & kw2) / len(kw1 | kw2)

def find_common_stories(threshold=0.18, top_n=5):
    """
    Fetch every source, group headlines that describe the same event,
    and rank those groups by how many distinct sources cover them.
    Most-covered story is first, least-covered (of the top N) is last.
    """
    all_stories = []
    sources = list(RSS_FEEDS.keys())
    progress = st.progress(0, text="Starting…")

    for i, source in enumerate(sources):
        progress.progress(i / len(sources), text=f"Scanning {source}…")
        for item in fetch_top_stories(source, limit=5):
            all_stories.append({
                "title": item["title"],
                "description": item.get("description", ""),
                "source": source,
                "link": item["link"],
                "keywords": _keywords(item["title"]),
            })
    progress.progress(1.0, text="Analyzing overlap…")

    # Clustering: drop each story into the BEST matching cluster (the one
    # with the highest similarity above the threshold), not just the first.
    clusters = []
    for story in all_stories:
        best_cluster = None
        best_score = threshold
        for cluster in clusters:
            score = _similarity(story["keywords"], cluster["keywords"])
            if score >= best_score:
                best_score = score
                best_cluster = cluster
        if best_cluster is not None:
            best_cluster["stories"].append(story)
            best_cluster["sources"].add(story["source"])
            best_cluster["keywords"] |= story["keywords"]
        else:
            clusters.append({
                "keywords": set(story["keywords"]),
                "stories": [story],
                "sources": {story["source"]},
            })

    # Pick the richest headline+description in each cluster as representative.
    for cluster in clusters:
        rep = max(cluster["stories"], key=lambda s: len(s["description"]))
        cluster["title"] = rep["title"]
        cluster["description"] = rep["description"]
        cluster["link"] = rep["link"]

    # Rank: most distinct sources first, then largest cluster.
    ranked = sorted(
        clusters,
        key=lambda c: (len(c["sources"]), len(c["stories"])),
        reverse=True,
    )
    progress.empty()
    return ranked[:top_n]

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

    /* Make bordered containers white to seamlessly blend iframe backgrounds */
    [data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #ffffff;
    }

    /* Ensure the sidebar keeps its native transparent background */
    [data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"] {
        background-color: transparent !important;
    }

    .accuracy-badge {
        float: right;
        background-color: #f1f5f9;
        color: #10b981;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 800;
        border: 1px solid #e2e8f0;
    }

    .model-badge {
        font-size: 12px; font-weight: 700; text-transform: uppercase;
        letter-spacing: 1px; margin-bottom: 10px; color: #64748b;
    }
    .meta-headline {
        font-size: 18px; font-weight: 800; color: #0f172a;
        line-height: 1.3; margin-bottom: 12px;
    }
    .meta-strapline {
        font-size: 14px; color: #ea580c; font-weight: 600;
        line-height: 1.4; border-left: 3px solid #ea580c;
        padding-left: 10px; margin-bottom: 12px;
    }
    .news-body { font-size: 14px; color: #334155; line-height: 1.6; word-wrap: break-word; }
    .source-banner {
        background: #e2e8f0; padding: 10px 15px; border-radius: 6px;
        font-size: 14px; font-weight: 600; color: #334155; margin-bottom: 15px;
    }

    /* Orange Publish button */
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
    }

    /* Orange regular button (Publish trigger) */
    div[data-testid="stButton"] > button[kind="secondary"] {
        background-color: #ea580c !important;
        color: #ffffff !important;
        border: none !important;
        font-weight: 700 !important;
        border-radius: 6px !important;
    }
    div[data-testid="stButton"] > button[kind="secondary"]:hover {
        background-color: #c2410c !important;
    }

    /* Black "Trending" button — high specificity so it overrides the
       generic secondary-button styling defined above. */
    .st-key-trending_btn button,
    .st-key-trending_btn button[kind="secondary"],
    div[data-testid="stButton"].st-key-trending_btn button {
        background-color: #000000 !important;
        color: #ffffff !important;
        border: none !important;
        font-weight: 700 !important;
        border-radius: 6px !important;
    }
    .st-key-trending_btn button:hover,
    .st-key-trending_btn button[kind="secondary"]:hover {
        background-color: #1f1f1f !important;
        color: #ffffff !important;
    }

    /* Ranked common-story card */
    .rank-card {
        background: #ffffff;
        border: 1px solid #cbd5e1;
        border-left: 5px solid #000000;
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 12px;
    }
    .rank-num {
        display: inline-block;
        background: #000000;
        color: #ffffff;
        font-weight: 800;
        font-size: 14px;
        width: 28px;
        height: 28px;
        line-height: 28px;
        text-align: center;
        border-radius: 50%;
        margin-right: 10px;
    }
    .rank-title { font-size: 16px; font-weight: 700; color: #0f172a; }
    .rank-meta  { font-size: 13px; color: #64748b; margin-top: 6px; }
    .rank-model {
        font-size: 11px; font-weight: 700; letter-spacing: 1px;
        color: #64748b; text-transform: uppercase; margin-top: 12px;
    }
    .rank-desc {
        font-size: 14px; color: #334155; line-height: 1.6;
        margin-top: 4px;
    }
    .rank-badge {
        display: inline-block; background: #ea580c; color: #fff;
        font-size: 12px; font-weight: 700; padding: 2px 10px;
        border-radius: 12px; margin-left: 8px;
    }
    </style>""", unsafe_allow_html=True)

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
    trending_clicked = st.button(
        "🔥 Trending",
        use_container_width=True,
        key="trending_btn",
    )
    st.write("### TRACKED SOURCES")
    selected_export_sources = []
    for source in list(RSS_FEEDS.keys()):
        col_left, col_right = st.columns([4, 1])
        with col_left:
            if st.button(source, use_container_width=True, key=f"btn_{source}"):
                st.session_state.active_source = source
        with col_right:
            if st.checkbox("Export", key=f"export_{source}", label_visibility="collapsed"):
                selected_export_sources.append(source)
    
    st.write("---")
    
    if st.session_state.publish_state == "ready":
        if st.session_state.trending_state == "ready":
            fname = f"newsdrum_trending_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx"
            label = "📥 Download Trending"
        else:
            fname = f"newsdrum_all_sources_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx"
            label = "📥 Download All"

        st.download_button(
            label=label,
            data=st.session_state.publish_excel,
            file_name=fname,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
    elif selected_export_sources:
        if st.button("📤 Compile Selected Sources", use_container_width=True, type="primary"):
            st.session_state.publish_state = "loading"
            st.rerun()

if st.session_state.active_source is None:
    st.session_state.active_source = list(RSS_FEEDS.keys())[0]
selected_source = st.session_state.active_source
if refresh_clicked:
    st.session_state.publish_state = "idle"
    st.session_state.publish_excel = None
    st.session_state.trending_state = "idle"   # return to normal feed
    st.session_state.live_feed_cache = {}      # Clear cache to force re-fetch

if trending_clicked:
    st.session_state.trending_state = "loading"
    st.rerun()

# ── Header row ────────────────────────────────────────────────
st.subheader(f"⚡ Live Feed: {selected_source}")

# ── PUBLISH: loading phase — fetch ALL sources + rewrite ──────
if st.session_state.publish_state == "loading":
    all_rows = []

    if st.session_state.trending_state == "ready" and st.session_state.trending_results:
        st.info("📤 Compiling trending stories for publish…")
        for cluster in st.session_state.trending_results:
            sources_str = ", ".join(sorted(cluster["sources"]))
            headline = cluster.get("ai_headline", cluster["title"])
            newsdrum_version = make_payload({
                "headline": headline,
                "strapline": cluster.get("ai_strapline", ""),
                "body": cluster.get("ai_description", "")
            })

            all_rows.append([
                f"Trending ({len(cluster['sources'])} sources)",
                cluster.get("link", "#"),
                cluster['title'],
                newsdrum_version
            ])
        st.session_state.publish_excel = build_excel(all_rows)
        st.session_state.publish_state = "ready"
        st.rerun()
    else:
        sources = selected_export_sources
        st.info("📤 Compiling selected sources for publish…")
        progress_bar = st.progress(0, text="Starting…")

        for i, source in enumerate(sources):
            progress_bar.progress((i) / len(sources), text=f"Fetching {source}…")
            
            if source in st.session_state.live_feed_cache and st.session_state.live_feed_cache[source] is not None:
                cached_source_data = st.session_state.live_feed_cache[source]
                for cached_item in cached_source_data:
                    item = cached_item["raw"]
                    data = cached_item["col1_data"]
                    all_rows.append([
                        source,
                        item['link'],
                        item['title'],
                        make_payload(data)
                    ])
            else:
                items = fetch_top_stories(source, limit=5)
                for item in items:
                    data = rewrite_with_gemini(item["title"], item["description"])
                    all_rows.append([
                        source,
                        item['link'],
                        item['title'],
                        make_payload(data)
                    ])

        progress_bar.progress(1.0, text="Done!")
        st.session_state.publish_excel = build_excel(all_rows)
        st.session_state.publish_state = "ready"
        st.rerun()
# ── TRENDING: loading phase — scan all sources for common stories ─
if st.session_state.trending_state == "loading":
    st.subheader("🔥 Trending Across All Sources")
    st.info("Scanning every source and measuring how widely each story is covered…")
    results = find_common_stories(threshold=0.18, top_n=5)

    # Generate a Gemini 2.5 Flash description for each trending story.
    if results:
        gen = st.progress(0, text="Writing summaries with Gemini 2.5 Flash…")
        for n, cluster in enumerate(results, 1):
            gen.progress((n - 1) / len(results),
                         text=f"Summarising story {n} of {len(results)}…")
            ai = rewrite_with_gemini(cluster["title"], cluster["description"])
            cluster["ai_headline"] = ai["headline"]
            cluster["ai_description"] = ai["body"]
            cluster["ai_strapline"] = ai["strapline"]
        gen.progress(1.0, text="Done!")
        gen.empty()

    st.session_state.trending_results = results
    st.session_state.trending_state = "ready"
    st.rerun()

# ── TRENDING: results view ────────────────────────────────────
if st.session_state.trending_state == "ready":
    head_l, head_r = st.columns([5, 1])
    with head_l:
        st.subheader("🔥 Trending Across All Sources")
    with head_r:
        if st.button("← Back to Feed", use_container_width=True):
            st.session_state.trending_state = "idle"
            st.rerun()

    results = st.session_state.trending_results
    if not results:
        st.warning("Couldn't gather enough stories to compare. Try again in a moment.")
    else:
        for rank, cluster in enumerate(results, 1):
            source_count = len(cluster["sources"])
            sources_list = ", ".join(sorted(cluster["sources"]))
            headline    = cluster.get("ai_headline") or cluster["title"]
            description = cluster.get("ai_description", "")
            strapline = cluster.get("ai_strapline", f"Covered by: {sources_list}")

            with st.container(border=True):
                col1, col2 = st.columns([5, 1])
                with col1:
                    st.markdown(
                        f'<div style="border-left: 5px solid #000000; padding-left: 15px;">'
                        f'<span class="rank-num">{rank}</span>'
                        f'<span class="rank-title">{headline}</span>'
                        f'<span class="rank-badge">{source_count} '
                        f'{"sources" if source_count != 1 else "source"}</span>'
                        f'<div class="rank-meta">📰 Covered by: {sources_list}</div>'
                        f'<div class="rank-model">GEMINI 2.5 FLASH</div>'
                        f'<div class="meta-strapline" style="margin-top: 10px;">{strapline}</div>'
                        f'<div class="rank-desc">{description}</div>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                with col2:
                    data = {"headline": headline, "strapline": strapline, "body": description}
                    payload = make_payload(data)
                    st_copy_to_clipboard(
                        payload,
                        before_copy_label="📋 Copy",
                        after_copy_label="✅ Copied!",
                        key=f"copy_trending_{rank}"
                    )

# ── Main live feed (hidden while publishing or viewing trending) ─
if st.session_state.publish_state != "loading" and st.session_state.trending_state == "idle":
    if selected_source:
        if selected_source not in st.session_state.live_feed_cache:
            with st.spinner(f"Intercepting top stories from {selected_source}..."):
                raw_items = fetch_top_stories(selected_source, limit=5)

            if not raw_items:
                st.session_state.live_feed_cache[selected_source] = None
            else:
                cached_items = []
                for idx, item in enumerate(raw_items):
                    with st.spinner(f"AI is writing story {idx + 1} of {len(raw_items)}…"):
                        col1_data = rewrite_with_gemini(item["title"], item["description"])
                        accuracy = calculate_accuracy_with_groq(item["description"], col1_data["body"])
                        # --- DISABLED: only Gemini news is shown for now ---
                        # col2_data = rewrite_with_nvidia(item["title"], item["description"])
                        # col3_data = rewrite_with_groq(item["title"], item["description"])
                    cached_items.append({
                        "raw": item,
                        "col1_data": col1_data,
                        "accuracy": accuracy,
                        # "col2_data": col2_data,
                        # "col3_data": col3_data
                    })
                st.session_state.live_feed_cache[selected_source] = cached_items

        cached_source_data = st.session_state.live_feed_cache.get(selected_source)

        if cached_source_data is None:
            st.error("Connection blocked by source firewall. Try a different outlet.")
        else:
            for idx, cached_item in enumerate(cached_source_data):
                item = cached_item["raw"]
                col1_data = cached_item["col1_data"]
                # --- DISABLED: only Gemini news is shown for now ---
                # col2_data = cached_item["col2_data"]
                # col3_data = cached_item["col3_data"]

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

                def render_native_card(model_label, data, column_ref, key_suffix, accuracy=None):
                    with column_ref:
                        with st.container(border=True):
                            acc_html = f'<div class="accuracy-badge">🎯 Accuracy: {accuracy}</div>' if accuracy else ''
                            st.markdown(f"""
                                {acc_html}
                                <div class="model-badge">{model_label}</div>
                                <div class="meta-headline">{data['headline']}</div>
                                <div class="meta-strapline">{data['strapline']}</div>
                                <div class="news-body">{data['body']}</div>
                                <br>
                            """, unsafe_allow_html=True)
                            payload = make_payload(data)
                            st_copy_to_clipboard(
                                payload,
                                before_copy_label="📋 Copy",
                                after_copy_label="✅ Copied!",
                                key=f"copy_{idx}_{key_suffix}"
                            )

                # Only Gemini is active — render it full width.
                render_native_card("Gemini 2.5 Flash", col1_data, st.container(), "1", accuracy=cached_item.get("accuracy"))
                # --- DISABLED: Groq and NVIDIA cards ---
                # col1, col2, col3 = st.columns(3)
                # render_native_card("NVIDIA Nemotron 1B",  col2_data, col2, "2")
                # render_native_card("Groq Llama 3.1",       col3_data, col3, "3")

# ── End of App ────────────────────────────────────────────────
