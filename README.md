# 📰 AI Journal — News-to-Audio Agent Powered by Ollama & Gemini

AI Journal turns real-time news into natural audio broadcasts using **LLMs + TTS**:

- Scrapes news headlines from the web
- Uses **Ollama (LLaMA3)** or **Gemini Pro** to summarize into a broadcast script
- Converts text to speech using **ElevenLabs**
- Returns an `.mp3` file for instant audio playback

---

## 🧠 Powered By

| Task              | Tool / API                 |
|-------------------|----------------------------|
| LLM Summarization | Ollama (LLaMA3) + Gemini   |
| News Scraping     | BrightData / Requests      |
| Text-to-Speech    | ElevenLabs                 |
| Backend           | FastAPI + LangGraph Agents |

---

## 🚀 Local Setup

### 1. Clone the repo

bash
git clone https://github.com/Rohit2200/Ai-journal.git
cd Ai-journal


2. Install dependencies
bash
pip install -r requirements.txt


3. Create .env


API_TOKEN=your_brightdata_token
WEB_UNLOCKER_ZONE=your_zone_id
GEMINI_API_KEY=your_google_key
ELEVENLABS_API_KEY=your_elevenlabs_key
No Anthropic key needed — Claude is not used anymore.

🧪 Run Locally
Start the FastAPI server:

bash

uvicorn backend:app --reload --port 1234
🔁 Sample Request

POST http://localhost:1234/generate-news-audio

Body:
{
  "topics": ["bitcoin"],
  "source_type": "news"
}

Returns: news-summary.mp3

Sample Output - https://drive.google.com/drive/folders/1YLj33bs_tld_Bwg3vkZx-oK31dCNbO9r?usp=sharing
