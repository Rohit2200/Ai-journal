import asyncio
import os
from typing import Dict, List
import httpx

from aiolimiter import AsyncLimiter
from tenacity import retry, stop_after_attempt, wait_exponential
from dotenv import load_dotenv

from utils import (
    clean_html_to_text,
    extract_headlines,
    summarize_with_anthropic_news_script,
    summarize_with_ollama
)

load_dotenv()

class NewsScraper:
    _rate_limiter = AsyncLimiter(5, 1)  # 5 requests per second
    serpapi_api_key = os.getenv("SERPAPI_API_KEY")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    async def scrape_news(self, topics: List[str]) -> Dict[str, str]:
        """Scrape and summarize news articles using SerpAPI"""
        results = {}

        for topic in topics:
            async with self._rate_limiter:
                try:
                    print(f"🔍 Searching news for topic: {topic}")
                    query_params = {
                        "q": topic,
                        "engine": "google",
                        "tbm": "nws", 
                        "api_key": self.serpapi_api_key
                    }

                    async with httpx.AsyncClient() as client:
                        response = await client.get("https://serpapi.com/search", params=query_params)
                        data = response.json()

                    articles = data.get("news_results", [])
                    if not articles:
                        results[topic] = f"No news articles found for '{topic}'."
                        continue

                    headlines = [article["title"] for article in articles[:5]]
                    headlines_text = "\n".join(headlines)

                    try:
                        summary = summarize_with_anthropic_news_script(
                            api_key=os.getenv("ANTHROPIC_API_KEY"),
                            headlines=headlines_text
                        )
                    except Exception:
                        summary = summarize_with_ollama(headlines_text)

                    results[topic] = summary

                except Exception as e:
                    results[topic] = f"Error: {str(e)}"

                await asyncio.sleep(1)  

        return {"news_analysis": results}
