from fastapi import FastAPI, HTTPException, File, Response
from fastapi.responses import FileResponse
import os
from pathlib import Path
from dotenv import load_dotenv

from models import NewsRequest
from utils import generate_broadcast_news, text_to_audio_elevenlabs_sdk, tts_to_audio
from news_scraper import NewsScraper
from reddit_scraper import scrape_reddit_topics

# Load environment variables
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '.env'))

print("API_TOKEN:", os.getenv("API_TOKEN"))
print("WEB_UNLOCKER_ZONE:", os.getenv("WEB_UNLOCKER_ZONE"))

app = FastAPI()

@app.post("/generate-news-audio")
async def generate_news_audio(request: NewsRequest):
    try:
        print("Received request:", request.dict())

        results = {}

        # News scraping
        if request.source_type in ["news", "both"]:
            print("Scraping news...")
            news_scraper = NewsScraper()
            results["news"] = await news_scraper.scrape_news(request.topics)
            print("News scraped:", results["news"])

        # Reddit scraping
        if request.source_type in ["reddit", "both"]:
            print("Scraping Reddit...")
            results["reddit"] = await scrape_reddit_topics(request.topics)
            print("Reddit scraped:", results["reddit"])

        # Combine sources
        news_data = results.get("news", {})
        reddit_data = results.get("reddit", {})

        print("Generating broadcast summary...")
        news_summary = generate_broadcast_news(
            news_data=news_data,
            reddit_data=reddit_data,
            topics=request.topics
        )
        print("Generated news summary:", news_summary)

        print("Converting to audio with ElevenLabs SDK...")
        audio_path = text_to_audio_elevenlabs_sdk(
            text=news_summary,
            voice_id="JBFqnCBsd6RMkjVDRZzb",
            model_id="eleven_multilingual_v2",
            output_format="mp3_44100_128",
            output_dir="audio"
        )
        print("Audio path:", audio_path)

        if audio_path and Path(audio_path).exists():
            with open(audio_path, "rb") as f:
                audio_bytes = f.read()

            return Response(
                content=audio_bytes,
                media_type="audio/mpeg",
                headers={"Content-Disposition": "attachment; filename=news-summary.mp3"}
            )
        else:
            raise HTTPException(status_code=500, detail="Audio file was not generated")

    except Exception as e:
        print("❌ ERROR:", str(e))
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend:app",
        host="0.0.0.0",
        port=1234,
        reload=True
    )
