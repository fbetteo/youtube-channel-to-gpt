"""
Get YouTube video publish dates via Innertube API.

Run interactively in IPython:
    %run scripts/test_innertube.py
    dates = await get_publish_dates(["VMj-3S1tku0", "dQw4w9WgXcQ", "jNQXAC9IVRw"])
"""

import asyncio
import os
import re
from urllib.parse import quote

import httpx

HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept-Language": "en-US,en;q=0.9",
}


def proxy_url():
    u = os.environ["WEBSHARE_PROXY_USERNAME"]
    p = os.environ["WEBSHARE_PROXY_PASSWORD"]
    return f"http://{quote(u)}-rotate:{quote(p)}@p.webshare.io:80"


async def get_publish_dates(video_ids, concurrency=50):
    # Fetch homepage directly (no proxy needed, just extracting API key)
    async with httpx.AsyncClient(
        headers=HEADERS, timeout=30, follow_redirects=True
    ) as direct:
        home = await direct.get("https://www.youtube.com")
        api_key = re.search(r'"INNERTUBE_API_KEY":"([^"]+)"', home.text).group(1)
        client_version = re.search(
            r'"INNERTUBE_CLIENT_VERSION":"([^"]+)"', home.text
        ).group(1)

    # Use proxy only for innertube player calls
    async with httpx.AsyncClient(
        headers=HEADERS, timeout=30, proxy=proxy_url(), follow_redirects=True
    ) as client:

        url = f"https://www.youtube.com/youtubei/v1/player?key={api_key}"
        sem = asyncio.Semaphore(concurrency)

        async def fetch(vid):
            payload = {
                "context": {
                    "client": {"clientName": "WEB", "clientVersion": client_version}
                },
                "videoId": vid,
            }
            async with sem:
                r = await client.post(url, json=payload)
                r.raise_for_status()
                data = r.json()
            date = (
                data.get("microformat", {})
                .get("playerMicroformatRenderer", {})
                .get("publishDate")
            )
            if not date:
                print(
                    f"{vid} -> no publishDate (status={data.get('playabilityStatus', {}).get('status')})"
                )
            return vid, date

        results = await asyncio.gather(*[fetch(v) for v in video_ids])
        return dict(results)


video_ids = ["VMj-3S1tku0"]
dates = await get_publish_dates(video_ids, concurrency=30)
dates
