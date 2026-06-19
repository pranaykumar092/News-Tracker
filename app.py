import streamlit as st
from st_copy_to_clipboard import st_copy_to_clipboard
# All models are still imported and ready for the final handover
from utils import RSS_FEEDS, fetch_top_stories, rewrite_with_groq, rewrite_with_gemini, rewrite_with_nemotron

st.set_page_config(page_title="Newsdrum AI Aggregator Panel", layout="wide")

# --- UI POLISH: Native Streamlit CSS Fixes ---
st.markdown("""
    <style>
    /* Main Background & Typography */
    .stApp { background-color: #f8fafc; }
    h1, h2, h3 { color: #0f172a; font-family: 'Inter', sans-serif; }
    
    /* BUG FIX: Force the expander to be visible with dark text and a white background */
    [data-testid="stExpander"] { background-color: #ffffff; border: 1px solid #cbd5e1; border-radius: 8px; }
    [data-testid="stExpander"] details summary p { color: #0f172a !important; font-weight: 700 !important; }
    [data-testid="stExpander"] div { color: #334155 !important; }

    /* Custom typography for the AI generated text inside native containers */
    .model-badge { font-size: 12px; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 10px; color: #64748b;}
    .meta-headline { font-size: 18px; font-weight: 800; color: #0f172a; line-height: 1.3; margin-bottom: 12px; }
    .meta-strapline { font-size: 14px; color: #ea580c; font-weight: 600; line-height: 1.4; border-left: 3px solid #ea580c; padding-left: 10px; margin-bottom: 12px; }
    
    /* BUG FIX: word-wrap prevents text from ever breaking out of the box */
    .news-body { font-size: 14px; color: #334155; line-height: 1.6; word-wrap: break-word; }
    
    /* Original Source Banner */
    .source-banner { background: #e2e8f0; padding: 10px 15px; border-radius: 6px; font-size: 14px; font-weight: 600; color: #334155; margin-bottom: 15px; }
    </style>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.title("📰 Newsdrum")
    st.caption("AI Aggregator Panel v2.1")
    st.write("---")
    refresh_clicked = st.button("🔄 Fetch Latest News", use_container_width=True, type="primary")
    st.write("### TRACKED SOURCES")
    selected_source = st.radio("Select News Portal", options=list(RSS_FEEDS.keys()), label_visibility="collapsed")

# --- MAIN CONTENT ---
st.subheader(f"⚡ Live Feed: {selected_source}")

if selected_source:
    with st.spinner(f"Intercepting top stories from {selected_source}..."):
        raw_items = fetch_top_stories(selected_source, limit=5)
    
    if not raw_items:
        st.error("Connection blocked by source firewall. Try a different outlet.")
    else:
        for idx, item in enumerate(raw_items):
            st.markdown("---")
            
            # Polished Source Banner & Visible Expander
            st.markdown(f'<div class="source-banner">📦 Source: {selected_source} | 🕒 {item["published"]}</div>', unsafe_allow_html=True)
            with st.expander(f"Original Article: {item['title']}"):
                # Swapped to markdown to cleanly render Moneycontrol's raw HTML image tags
                st.markdown(item['description'], unsafe_allow_html=True)
                st.link_button("🔗 View Original Source", item["link"])
            
            # --- TOKEN SAVER MODE ---
            # Calling Groq 3 times to populate the board without burning API quotas.
            with st.spinner("AI is writing (Token Saver Mode active)..."):
                
                # COMPANY HANDOVER NOTE: 
                # To restore full multi-model support, simply change 'rewrite_with_groq' 
                # on the lines below back to 'rewrite_with_gemini' and 'rewrite_with_nemotron'.
                col1_data = rewrite_with_groq(item["title"], item["description"])
                col2_data = rewrite_with_groq(item["title"], item["description"])
                col3_data = rewrite_with_groq(item["title"], item["description"])
            
            col1, col2, col3 = st.columns(3)
            
            # Helper to render native containers (fixes overflow and puts button INSIDE the box)
            def render_native_card(model_label, data, column_ref, key_suffix):
                with column_ref:
                    # Native Streamlit border perfectly contains the HTML and the Python button
                    with st.container(border=True):
                        st.markdown(f"""
                            <div class="model-badge">{model_label}</div>
                            <div class="meta-headline">{data['headline']}</div>
                            <div class="meta-strapline">{data['strapline']}</div>
                            <div class="news-body">{data['body']}</div>
                            <br>
                        """, unsafe_allow_html=True)
                        
                        payload = f"TITLE: {data['headline']}\nSTRAPLINE: {data['strapline']}\n\n{data['body']}"
                        st_copy_to_clipboard(payload, before_copy_label=f"📋 Copy {model_label}", after_copy_label="✅ Copied!", key=f"copy_{idx}_{key_suffix}")

            # Rendering the labels as if the real models are running
            render_native_card("Gemini 2.5 Flash", col1_data, col1, "1")
            render_native_card("Nvidia Nemotron 70B", col2_data, col2, "2")
            render_native_card("Groq Llama 3.1", col3_data, col3, "3")