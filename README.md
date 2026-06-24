# 📰 Newsdrum AI Aggregator Panel

A **Streamlit-powered news aggregator** that pulls live RSS feeds from major Indian and global news outlets and rewrites each story using AI models (Groq, Gemini, Cerebras) in the distinct Newsdrum editorial style.

---

## ✨ Features

- 🔴 **Live RSS feeds** from NDTV, Moneycontrol, Indian Express, Economic Times, The Hindu, Times of India, Hindustan Times, and AP News
- 🤖 **AI-powered rewriting** using Groq (Llama 3.1), Google Gemini 2.5 Flash, and Cerebras
- 📋 **One-click copy** for each AI-generated article
- ⚡ Clean, minimal editorial UI built with Streamlit

---

## 🛠️ Prerequisites

Make sure you have the following installed before starting:

- **Python 3.9+** → [Download Python](https://www.python.org/downloads/)
- **pip** (comes bundled with Python)
- **Git** → [Download Git](https://git-scm.com/)

---

## 🚀 Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/pranaykumar092/News-Tracker.git
cd News-Tracker
```

### 2. Create a Virtual Environment (Recommended)

```bash
# Create the virtual environment
python -m venv venv

# Activate it — Windows
venv\Scripts\activate

# Activate it — macOS / Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🔐 Setting Up Your `.env` File (Required)

The app requires API keys from three providers. You **must** create a `.env` file in the root of the project folder before running the app.

### Step 1 — Create the file

In the project root (`News-Tracker/`), create a new file named exactly **`.env`** (note the leading dot, no other extension).

```
News-Tracker/
├── .env          ← Create this file here
├── app.py
├── utils.py
├── requirements.txt
└── ...
```

### Step 2 — Add your API keys

Open `.env` and paste the following, replacing each placeholder with your actual key:

```env
GROQ_API_KEY=your_groq_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here
CEREBRAS_API_KEY=your_cerebras_api_key_here
```

### Step 3 — Get your API keys

| Key | Provider | Get it here |
|---|---|---|
| `GROQ_API_KEY` | Groq (free tier available) | [console.groq.com](https://console.groq.com) → API Keys |
| `GEMINI_API_KEY` | Google AI Studio (free tier available) | [aistudio.google.com](https://aistudio.google.com) → Get API Key |
| `CEREBRAS_API_KEY` | Cerebras | [cloud.cerebras.ai](https://cloud.cerebras.ai/) → API Keys |

> ⚠️ **Never commit your `.env` file to GitHub.** It is already listed in `.gitignore` and will be ignored automatically.

---

## ▶️ Running the App

Once your virtual environment is active and your `.env` file is in place, run:

```bash
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`.

---

## 📁 Project Structure

```
News-Tracker/
│
├── app.py              # Main Streamlit UI
├── utils.py            # RSS fetching logic & AI model functions
├── requirements.txt    # Python dependencies
├── .gitignore          # Files excluded from git (including .env)
└── .env                # Your local API keys — NOT committed to GitHub
```

---

## 🔧 Troubleshooting

**App starts but AI cards show an error message?**
→ Double-check your `.env` file exists in the project root and that all three API keys are correctly pasted with no extra spaces.

**`ModuleNotFoundError` when running?**
→ Make sure your virtual environment is activated (`venv\Scripts\activate` on Windows) and that you ran `pip install -r requirements.txt`.

**RSS feed shows "Connection blocked by source firewall"?**
→ Some news sources block automated requests intermittently. Try selecting a different news outlet from the sidebar and refresh.

**Port already in use?**
→ Run on a different port: `streamlit run app.py --server.port 8502`

---

## 📄 License

This project is for internal/educational use. All news content belongs to their respective original publishers.
