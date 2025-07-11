import httpx
from loguru import logger
from typing import Optional, Dict, Any
import time
import asyncio
import random

def safe_sync_get(
    session: httpx.Client,
    url: str,
    params: dict | None = None,
    max_attempts: int = 5,
    rate_limit: float = 3.0,
) -> Optional[Dict[str, Any]]:
    backoff = 1.0
    for attempt in range(1, max_attempts + 1):
        try:
            resp = session.get(url, params=params)
            resp.raise_for_status()
            time.sleep(rate_limit)
            return resp.json()
        except (httpx.ReadTimeout, httpx.HTTPStatusError) as e:
            code = getattr(e.response, "status_code", None)
            logger.warning(f"[{attempt}/{max_attempts}] HTTP {code} on {url}; {'timeout' if isinstance(e, httpx.ReadTimeout) else 'error'}.")
            if attempt == max_attempts:
                logger.error(f"Giving up on {url} after {max_attempts} attempts.")
                return None if code and code >= 400 else {}
            time.sleep(backoff)
            backoff = min(backoff * 2, 10)
    return None

async def safe_async_get(
    client: httpx.AsyncClient,
    url: str,
    params: dict | None = None,
    max_attempts: int = 5,
    rate_limit: float = 3.0,
    semaphore: asyncio.Semaphore | None = None,
) -> Optional[Dict[str, Any]]:
    backoff = 1.0
    sem = semaphore or asyncio.Semaphore(1)
    for attempt in range(1, max_attempts + 1):
        async with sem:
            try:
                resp = await client.get(url, params=params)
                resp.raise_for_status()
                await asyncio.sleep(rate_limit + random.random() * 0.5)
                return resp.json()
            except (httpx.ReadTimeout, httpx.HTTPStatusError) as e:
                code = getattr(e.response, "status_code", None)
                logger.warning(f"[{attempt}/{max_attempts}] HTTP {code} on {url}; error.")
                if attempt == max_attempts:
                    logger.error(f"Giving up on {url} after {max_attempts} attempts.")
                    return None
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 10)
    return None 