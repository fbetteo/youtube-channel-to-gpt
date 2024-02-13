import functools
import pickle

from db.test_ps import connection, cache
from fastapi import HTTPException


def load_assistant(func):
    # check if assistant is in cache, if not, load it from db
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        user_id = kwargs.get("user_id")
        assistant_name = kwargs.get("assistant_name")

        if not cache.get(user_id, {}).get(assistant_name, {}).get("assistant"):
            c = connection.cursor()
            c.execute(
                f"SELECT assistant_id FROM assistants WHERE user_id = {user_id} and assistant_name = '{assistant_name}'"
            )
            try:
                assistant_id = c.fetchone()[0]
            except:
                # esta bien hacerlo asi?
                print("Assistant not found")
                # raise Exception("Assistant not found")
                raise HTTPException(status_code=404, detail="Assistant not found")
            c.close()

            try:
                with open(f"{user_id}_{assistant_id}.pkl", "rb") as f:
                    channel_assistant = pickle.load(f)
                cache[user_id][assistant_name] = {}
                cache[user_id][assistant_name]["assistant"] = channel_assistant
            except:
                raise Exception("Assistant exists but couldn't be loaded")

        value = func(*args, **kwargs)
        # return value

    return wrapper
