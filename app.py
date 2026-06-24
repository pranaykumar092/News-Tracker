import streamlit as st
import io
from datetime import datetime
from openpyxl import Workbook
from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
from openpyxl.utils import get_column_letter
from st_copy_to_clipboard import st_copy_to_clipboard
from utils import RSS_FEEDS, fetch_top_stories, rewrite_with_groq, rewrite_with_gemini, rewrite_with_nemotron

st.set_page_config(page_title="Newsdrum AI Aggregator Panel", layout="wide")



def build_excel(rows):
    """
    Build an xlsx file with:
    - Bold header row
    - Text wrap + top-align on every cell
    - Column widths auto-fitted to content (capped at 80 chars)
    - Row heights scaled to content length
    """
    wb = Workbook()
    ws = wb.active
    ws.title = "Newsdrum Export"

    headers = ["Source News", "News Headline", "News Description", "Newsdrum Version"]
    col_widths = [len(h) for h in headers]

    # ── Thin border ───────────────────────────────────────────
    def thin():
        s = Side(style="thin", color="CCCCCC")
        return Border(left=s, right=s, top=s, bottom=s)

    # ── Header row ────────────────────────────────────────────
    for c, h in enumerate(headers, 1):
        cell = ws.cell(row=1, column=c, value=h)
        cell.font      = Font(name="Arial", bold=True, size=11, color="FFFFFF")
        cell.fill      = PatternFill("solid", start_color="1F2D3D")
        cell.alignment = Alignment(horizontal="center", vertical="center",
                                   wrap_text=True)
        cell.border    = thin()
    ws.row_dimensions[1].height = 22

    # ── Data rows ─────────────────────────────────────────────
    for r, row in enumerate(rows, 2):
        max_lines = 1
        for c, val in enumerate(row, 1):
            text = str(val) if val else ""
            cell = ws.cell(row=r, column=c, value=text)
            cell.font      = Font(name="Arial", size=10)
            cell.alignment = Alignment(vertical="top", wrap_text=True)
            cell.border    = thin()

            # Track widest content per column (cap at 80)
            longest_line = max((len(line) for line in text.split("\n")), default=0)
            col_widths[c - 1] = min(80, max(col_widths[c - 1], longest_line))

            # Estimate how many lines this cell will wrap to (at col width chars)
            cap = col_widths[c - 1] if col_widths[c - 1] > 0 else 1
            lines = sum(
                max(1, (len(line) + cap - 1) // cap)
                for line in text.split("\n")
            )
            max_lines = max(max_lines, lines)

        # Row height: ~15pt per wrapped line, min 18, max 400
        ws.row_dimensions[r].height = min(400, max(18, max_lines * 15))

    # ── Apply column widths ───────────────────────────────────
    for i, w in enumerate(col_widths, 1):
        ws.column_dimensions[get_column_letter(i)].width = w + 4  # +4 padding

    # ── Freeze header row ─────────────────────────────────────
    ws.freeze_panes = "A2"

    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)
    return buf.getvalue()

def render_export_button(slot):
    has_data = len(st.session_state.csv_rows) > 0

    with slot:
        if has_data:
            count = len(st.session_state.csv_rows)
            fname = (
                f"newsdrum_"
                f"{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx"
            )
            excel_bytes = build_excel(st.session_state.csv_rows)
            st.download_button(
                label=f"📤 Publish ({count})",
                data=excel_bytes,
                file_name=fname,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
        else:
            st.button(
                "📤 Publish",
                disabled=True,
                use_container_width=True,
                help="Fetch news first to enable publish."
            )

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

    /* Orange Export button */
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

# ── Track active source; only wipe rows on manual refresh ────
st.session_state.active_source = selected_source
if refresh_clicked:
    st.session_state.csv_rows = []

# ── Header row: title left, button placeholder right ─────────
header_col, btn_col = st.columns([5, 1])
with header_col:
    st.subheader(f"⚡ Live Feed: {selected_source}")

btn_slot = btn_col.empty()   # filled after the news loop

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

            # Accumulate for export — guard against duplicates on rerun
            already_added = any(
                row[0] == selected_source and row[1] == col1_data["headline"]
                for row in st.session_state.csv_rows
            )
            if not already_added:
                def make_payload(data):
                    return (
                        f"TITLE: {data['headline']}\n"
                        f"STRAPLINE: {data['strapline']}\n\n"
                        f"{data['body']}"
                    )
                st.session_state.csv_rows.append([
                    selected_source,            # Source News (e.g. NDTV, Moneycontrol)
                    col1_data["headline"],      # News Headline — Gemini 2.5 Flash
                    col1_data["body"],          # News Description — Gemini 2.5 Flash
                    make_payload(col1_data),    # Newsdrum Version — full TITLE + STRAPLINE + BODY
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

            render_native_card("Gemini 2.5 Flash",    col1_data, col1, "1")
            render_native_card("Nvidia Nemotron 70B",  col2_data, col2, "2")
            render_native_card("Groq Llama 3.1",       col3_data, col3, "3")

# ── Fill button slot now that csv_rows is fully populated ─────
render_export_button(btn_slot)
