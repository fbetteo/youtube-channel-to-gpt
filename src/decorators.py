import functools
import pickle
from db.database import get_connection, get_cache
from fastapi import HTTPException
import logging


def load_assistant(func):
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        # Expect user_id and assistant_name are passed; get cache from dependency
        user_id = kwargs.get("user_id")
        assistant_name = kwargs.get("assistant_name")
        cache = get_cache()  # use injected cache provider
        if not cache.get(user_id, {}).get(assistant_name, {}).get("assistant"):
            db = get_connection()
            c = db.cursor()
            c.execute(
                "SELECT assistant_id FROM assistants WHERE user_id = %s AND assistant_name = %s",
                (user_id, assistant_name),
            )
            try:
                assistant_id = c.fetchone()[0]
            except:
                logging.error(
                    "Assistant not found for user %s with assistant %s",
                    user_id,
                    assistant_name,
                )
                raise HTTPException(status_code=404, detail="Assistant not found")
            c.close()
            db.close()
            try:
                with open(f"{user_id}_{assistant_id}.pkl", "rb") as f:
                    channel_assistant = pickle.load(f)
                cache.setdefault(user_id, {})[assistant_name] = {
                    "assistant": channel_assistant
                }
            except:
                raise Exception("Assistant exists but couldn't be loaded")
        return await func(*args, **kwargs)

    return wrapper


def load_assistant_returning_object(func):
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        user_id = kwargs.get("user_id")
        assistant_name = kwargs.get("assistant_name")
        cache = get_cache()
        if not cache.get(user_id, {}).get(assistant_name, {}).get("assistant"):
            db = get_connection()
            c = db.cursor()
            c.execute(
                "SELECT assistant_id FROM assistants WHERE user_id = %s AND assistant_name = %s",
                (user_id, assistant_name),
            )
            try:
                assistant_id = c.fetchone()[0]
            except:
                logging.error(
                    "Assistant not found for user %s with assistant %s",
                    user_id,
                    assistant_name,
                )
                raise HTTPException(status_code=404, detail="Assistant not found")
            c.close()
            db.close()
            try:
                with open(f"{user_id}_{assistant_id}.pkl", "rb") as f:
                    channel_assistant = pickle.load(f)
                cache.setdefault(user_id, {})[assistant_name] = {
                    "assistant": channel_assistant
                }
            except:
                raise Exception("Assistant exists but couldn't be loaded")
        value = await func(*args, **kwargs)
        return value

    return wrapper
