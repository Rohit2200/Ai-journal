from typing import List
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
import asyncpraw

load_dotenv()

# Initialize Claude
model = ChatAnthropic(model="claude-3-5-sonnet-20240620")

# Setup async Reddit client
reddit = asyncpraw.Reddit(
    client_id=os.getenv("REDDIT_CLIENT_ID"),
    client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
    user_agent=os.getenv("REDDIT_USER_AGENT")
)

two_weeks_ago = datetime.utcnow() - timedelta(days=14)

async def fetch_top_reddit_posts(topic: str, limit: int = 5):
    posts = []
    try:
        subreddit = await reddit.subreddit("all")
        async for submission in subreddit.search(topic, sort="top", limit=limit, time_filter="month"):
            post_time = datetime.utcfromtimestamp(submission.created_utc)
            if post_time >= two_weeks_ago:
                posts.append({
                    "title": submission.title,
                    "selftext": submission.selftext[:500], 
                    "score": submission.score,
                    "url": submission.url
                })
    except Exception as e:
        print(f"Error fetching posts for topic {topic}: {e}")
    return posts


async def analyze_posts_with_llm(topic: str, posts: List[dict]):
    formatted_posts = "\n\n".join([
        f"Title: {p['title']}\nContent: {p['selftext']}\nScore: {p['score']}" for p in posts
    ])

    prompt = f"""
You are a Reddit analysis expert. Below are Reddit posts about '{topic}' from the last 2 weeks:

{formatted_posts}

Please provide:
1. Main discussion points
2. Key opinions expressed
3. Notable trends or patterns
4. Quotes from interesting comments (no usernames)
5. Overall sentiment (positive/neutral/negative)
    """

    response = await model.ainvoke(prompt)
    return response.content.strip()


async def scrape_reddit_topics(topics: List[str]) -> dict[str, dict]:
    reddit_results = {}

    for topic in topics:
        posts = await fetch_top_reddit_posts(topic)
        if posts:
            summary = await analyze_posts_with_llm(topic, posts)
            reddit_results[topic] = summary
        else:
            reddit_results[topic] = "No recent posts found or error occurred."

    return {"reddit_analysis": reddit_results}
