from urllib.parse import quote_plus
from dotenv import load_dotenv
import requests
import os
from fastapi import FastAPI, HTTPException
from bs4 import BeautifulSoup
import ollama
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage
from datetime import datetime
from elevenlabs import ElevenLabs
import google.generativeai as genai  # ✅ Added for Gemini

load_dotenv()

class MCPOverloadedError(Exception):
    """Custom exception for MCP service overloads"""
    pass


def generate_valid_news_url(keyword: str) -> str:
    q = quote_plus(keyword)
    return f"https://news.google.com/search?q={q}&tbs=sbd:1"


def generate_news_urls_to_scrape(list_of_keywords):
    valid_urls_dict = {}
    for keyword in list_of_keywords:
        valid_urls_dict[keyword] = generate_valid_news_url(keyword)
    return valid_urls_dict


def scrape_with_brightdata(url: str) -> str:
    headers = {
        "Authorization": f"Bearer {os.getenv('BRIGHTDATA_API_KEY')}",
        "Content-Type": "application/json"
    }

    payload = {
        "zone": os.getenv('BRIGHTDATA_WEB_UNLOCKER_ZONE'),
        "url": url,
        "format": "raw"
    }

    try:
        response = requests.post("https://api.brightdata.com/request", json=payload, headers=headers)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"BrightData error: {str(e)}")


def clean_html_to_text(html_content: str) -> str:
    soup = BeautifulSoup(html_content, "html.parser")
    text = soup.get_text(separator="\n")
    return text.strip()


def extract_headlines(cleaned_text: str) -> str:
    headlines = []
    current_block = []
    lines = [line.strip() for line in cleaned_text.split('\n') if line.strip()]

    for line in lines:
        if line == "More":
            if current_block:
                headlines.append(current_block[0])
                current_block = []
        else:
            current_block.append(line)

    if current_block:
        headlines.append(current_block[0])

    return "\n".join(headlines)


def summarize_with_ollama(headlines) -> str:
    prompt = f"""You are my personal news editor. Summarize these headlines into a TV news script for me, focus on important headlines and remember that this text will be converted to audio:
    So no extra stuff other than text which the podcaster/newscaster should read, no special symbols or extra information in between and of course no preamble please.
    {headlines}
    News Script:"""

    try:
        client = ollama.Client(host=os.getenv("OLLAMA_HOST", "http://localhost:11434"))
        response = client.generate(
            model="llama3",
            prompt=prompt,
            options={
                "temperature": 0.4,
                "max_tokens": 800
            },
            stream=False
        )
        return response['response']
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ollama error: {str(e)}")


# ✅ REPLACED THIS FUNCTION TO USE GEMINI
def generate_broadcast_news(news_data, reddit_data, topics):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is missing in environment")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")

    topic_blocks = []
    for topic in topics:
        news_content = news_data["news_analysis"].get(topic) if news_data else ''
        reddit_content = reddit_data["reddit_analysis"].get(topic) if reddit_data else ''
        context = []
        if news_content:
            context.append(f"OFFICIAL NEWS CONTENT:\n{news_content}")
        if reddit_content:
            context.append(f"REDDIT DISCUSSION CONTENT:\n{reddit_content}")
        if context:
            topic_blocks.append(
                f"TOPIC: {topic}\n\n" + "\n\n".join(context)
            )

    user_prompt = (
        "You are a professional broadcast journalist. Create a 3–4 paragraph news script from the following:\n\n"
        + "\n\n--- NEW TOPIC ---\n\n".join(topic_blocks)
        + "\n\nInstructions:\n"
        "- Avoid intro phrases or extra narration\n"
        "- Write natural spoken paragraphs\n"
        "- Use short quotes from Reddit if helpful\n"
        "- Maintain neutral tone\n"
        "- End with 'To wrap up this segment...'"
    )

    try:
        response = model.generate_content(user_prompt)
        return response.text.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini error: {str(e)}")


def summarize_with_anthropic_news_script(api_key: str, headlines: str) -> str:
    system_prompt = """
You are my personal news editor and scriptwriter for a news podcast. Your job is to turn raw headlines into a clean, professional, and TTS-friendly news script.

The final output will be read aloud by a news anchor or text-to-speech engine. So:
- Do not include any special characters, emojis, formatting symbols, or markdown.
- Do not add any preamble or framing like "Here's your summary" or "Let me explain".
- Write in full, clear, spoken-language paragraphs.
- Keep the tone formal, professional, and broadcast-style — just like a real TV news script.
- Focus on the most important headlines and turn them into short, informative news segments that sound natural when spoken.
- Start right away with the actual script, using transitions between topics if needed.
"""

    try:
        llm = ChatAnthropic(
            model="claude-3-opus-20240229",
            api_key=api_key,
            temperature=0.4,
            max_tokens=1000
        )
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=headlines)
        ])
        return response.content
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Anthropic error: {str(e)}")


def text_to_audio_elevenlabs_sdk(
    text: str,
    voice_id: str = "JBFqnCBsd6RMkjVDRZzb",
    model_id: str = "eleven_multilingual_v2",
    output_format: str = "mp3_44100_128",
    output_dir: str = "audio",
    api_key: str = None
) -> str:
    try:
        api_key = api_key or os.getenv("ELEVEN_API_KEY")
        if not api_key:
            raise ValueError("ElevenLabs API key is required.")
        client = ElevenLabs(api_key=api_key)
        audio_stream = client.text_to_speech.convert(
            text=text,
            voice_id=voice_id,
            model_id=model_id,
            output_format=output_format
        )
        os.makedirs(output_dir, exist_ok=True)
        filename = f"tts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp3"
        filepath = os.path.join(output_dir, filename)
        with open(filepath, "wb") as f:
            for chunk in audio_stream:
                f.write(chunk)
        return filepath
    except Exception as e:
        raise e


from pathlib import Path
from gtts import gTTS
AUDIO_DIR = Path("audio")
AUDIO_DIR.mkdir(exist_ok=True)

def tts_to_audio(text: str, language: str = 'en') -> str:
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = AUDIO_DIR / f"tts_{timestamp}.mp3"
        tts = gTTS(text=text, lang=language, slow=False)
        tts.save(str(filename))
        return str(filename)
    except Exception as e:
        print(f"gTTS Error: {str(e)}")
        return None
