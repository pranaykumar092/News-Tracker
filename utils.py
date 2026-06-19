import os
import feedparser
from groq import Groq
import google.generativeai as genai
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Configure APIs
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Direct official Nvidia client connection
nvidia_client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.environ.get("NVIDIA_API_KEY")
)

# 1. RSS Feed Sources configuration
RSS_FEEDS = {
    "NDTV": "https://feeds.feedburner.com/ndtvnews-top-stories",
    "Moneycontrol": "https://www.moneycontrol.com/rss/MCtopnews.xml",
    "Indian Express": "https://indianexpress.com/section/india/feed/",
    "Economic Times": "https://economictimes.indiatimes.com/rssfeedstopstories.cms",
    "The Hindu": "https://www.thehindu.com/news/national/feeder/default.rss",
    "Times of India": "https://timesofindia.indiatimes.com/rssfeedstopstories.cms",
    "Hindustan Times": "https://www.hindustantimes.com/feeds/rss/topnews/rssfeed.xml",
    "AP News": "https://apnews.com/feed"
}

# 2. Global System Prompt for all models
SYSTEM_PROMPT = (
    "You are an expert news editor working for newsdrum.in. Your task is to rewrite incoming news "
    "articles into the distinct Newsdrum style.\n\n"
    "Strict Editorial Guidelines:\n"
    "1. VOICE: Write in simple, crystal-clear English. Be factual, straightforward, blunt, and "
    "not politically correct. View every event through a sharp Indian point of view.\n"
    "2. HEADLINE: Create a powerful, direct, engaging, and slightly clickbaity headline.\n"
    "3. STRAPLINE: Create an exact 25-word strapline directly under the headline.\n"
    "4. BODY LENGTH: The body must be strictly between 75 and 100 words. Do not write less than 75 words. Do not exceed 100 words.\n\n"
    "Format exactly like this, with double line breaks:\n"
    "HEADLINE: [Headline]\n\n"
    "STRAPLINE: [25-word Strapline]\n\n"
    "BODY: [75-100 word body]"
)

def fetch_top_stories(source_name, limit=5):
    feed_url = RSS_FEEDS.get(source_name)
    if not feed_url: 
        return []
    
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/121.0.0.0 Safari/537.36"}
    parsed_feed = feedparser.parse(feed_url, agent=headers["User-Agent"])
    
    stories = []
    for entry in parsed_feed.entries[:limit]:
        stories.append({
            "title": entry.get("title", "No Headline Available"),
            "description": entry.get("summary", entry.get("description", "No context available.")),
            "link": entry.get("link", "#"),
            "published": entry.get("published", "Just now")
        })
    return stories

def parse_ai_response(response_text, original_title):
    """Helper to cleanly split the AI output into a dictionary with robust fallbacks."""
    if not response_text:
        return {"headline": original_title, "strapline": "API Error", "body": "The model failed to generate text."}

    lines = response_text.split("\n")
    parsed = {"headline": "", "strapline": "", "body": ""}
    
    reading_body = False 
    body_content = []

    for line in lines:
        clean_line = line.replace("**", "").strip()
        if not clean_line:
            continue
            
        check_line = clean_line.upper()
        
        if check_line.startswith("HEADLINE:") or check_line.startswith("HEADLINE :"):
            parsed["headline"] = clean_line.split(":", 1)[1].strip().strip('"').strip("'")
            reading_body = False
            
        elif check_line.startswith("STRAPLINE:") or check_line.startswith("STRAPLINE :"):
            parsed["strapline"] = clean_line.split(":", 1)[1].strip()
            reading_body = False
            
        elif check_line.startswith("BODY:") or check_line.startswith("BODY :"):
            body_text = clean_line.split(":", 1)[1].strip()
            if body_text:
                body_content.append(body_text)
            reading_body = True
            
        elif reading_body:
            body_content.append(clean_line)
    
    parsed["body"] = "\n\n".join(body_content).strip()

    # --- THE ULTIMATE FALLBACKS ---
    # 1. If the AI forgot the headline, use the original RSS article title!
    if not parsed["headline"] or parsed["headline"] == "Generated Output":
        parsed["headline"] = original_title

    # 2. If it missed the strapline, provide a clean default instead of an error message
    if not parsed["strapline"] or parsed["strapline"] == "Formatting tags missing":
        parsed["strapline"] = "AI Summary generated from original source text."

    # 3. If the body is empty because tags were completely missing, dump everything into the body
    if not parsed["body"]:
        parsed["body"] = response_text

    return parsed

def rewrite_with_groq(title, context):
    try:
        chat = groq_client.chat.completions.create(
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": f"Title: {title}\nContext: {context}"}],
            model="llama-3.1-8b-instant", 
            temperature=0.7, 
            max_tokens=500
        )
        return parse_ai_response(chat.choices[0].message.content, title)
    except Exception as e: 
        return {"headline": title, "strapline": "Groq Error", "body": str(e)}

def rewrite_with_gemini(title, context):
    try:
        model = genai.GenerativeModel('gemini-2.5-flash', system_instruction=SYSTEM_PROMPT)
        response = model.generate_content(f"Title: {title}\nContext: {context}")
        return parse_ai_response(response.text, title)
    except Exception as e: 
        return {"headline": title, "strapline": "Gemini Error", "body": str(e)}

def rewrite_with_nemotron(title, context):
    try:
        chat = nvidia_client.chat.completions.create(
            model="meta/llama-3.1-nemotron-70b-instruct", 
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": f"Title: {title}\nContext: {context}"}],
            temperature=0.7, 
            max_tokens=500
        )
        return parse_ai_response(chat.choices[0].message.content, title)
    except Exception as e: 
        return {"headline": title, "strapline": "Nvidia Error", "body": str(e)}