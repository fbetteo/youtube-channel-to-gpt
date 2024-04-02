import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import pickle
from typing import List, Union
import fastapi_retrieve
import fastapi_assistant
from fastapi import FastAPI, Body, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from db.test_ps import connection, cache
from db.models import User
import json
from decorators import load_assistant, load_assistant_returning_object
import utils
import uuid


from fastapi import HTTPException, Depends, Header, Security, Cookie
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from typing import Optional
from supabase import create_client, Client

app = FastAPI()

url: str = os.getenv("NEXT_PUBLIC_SUPABASE_URL")
anon_key: str = os.getenv("SUPABASE_ANON_KEY")
secret_key: str = os.getenv("SUPABASE_SECRET")
supabase: Client = create_client(url, anon_key)


ALGORITHM = "HS256"


print(ALGORITHM)
# def query_supabase_as_user(jwt: str):
#     # Replace the anon key with the user's JWT
#     supabase.headers = {
#         "Authorization": f"Bearer {jwt}",
#         "apikey": "YOUR_SUPABASE_ANON_KEY",
#     }

#     # Example query
#     data = supabase.table("your_table").select("*").execute()
#     return data


# I think this was needed to allow the frontend to connect to the backend
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "localhost:3000",
    "http://localhost:3000/",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

## I create global cache and connectoin in db/test_ps.py (cache could be somewhere else really..) so I can call them from decorators.py and from here
## The decorator will use cache and connection to check if the assistant is in cache or not, and if not, it will load it from the pickle.
## TODO: double check if this works and implement for the other methods
## TODO: implement db update with threads and messages


# global cache
# cache = {}
# # 1 is user_id: 1
# # cache[1] = {}

security = HTTPBearer()


async def validate_jwt(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """
    Extracts the JWT from the cookie and validates it.
    Returns the payload if the token is valid, raises an HTTPException otherwise.
    """
    token = credentials.credentials
    print(token)
    if token is None:
        raise HTTPException(status_code=401, detail="JWT token is missing")

    try:
        payload = jwt.decode(
            token, secret_key, algorithms=[ALGORITHM], options={"verify_aud": False}
        )
        return (
            payload  # Or extract specific data you need, e.g., user_id: payload["sub"]
        )
    except JWTError:
        raise HTTPException(status_code=403, detail="Could not validate credentials")


@app.get("/")
def read_root():
    return {"Hello": "World"}


# Deberiamos tener una sola funcion cuando el usuario quiera generar un assistant, que baje el transcript y cree el assistant. Asi no guardamos en ningun momento el transcript?
@app.get("/transcripts/{assistant_name}")
def get_channel_transcript(
    assistant_name: str, channel_id: str, max_results: int, user_id: uuid.UUID
):

    print("hello")
    print(user_id)
    video_retrieval = fastapi_retrieve.VideoRetrieval(channel_id, max_results)
    video_retrieval.get_video_ids()
    video_retrieval.get_transcripts()
    if cache.get(user_id) is None:
        cache[user_id] = {}
    cache[user_id][assistant_name] = {}
    cache[user_id][assistant_name]["video_retrieval"] = video_retrieval
    print(video_retrieval.all_transcripts[0:10])


@app.get("/assistants/{user_id}")
def get_assistants(user_id: int, uuid: uuid.UUID):
    output = []
    c = connection.cursor()
    c.execute(
        f"SELECT assistant_id, assistant_name FROM assistants WHERE user_id = {user_id} and uuid = '{uuid}'"
    )
    # print(c.fetchall())
    results = c.fetchall()
    c.close()

    for row in results:
        output.append({"id": row[0], "name": row[1]})

    return output


@app.get("/assistants-protected")
def get_assistants_protected(
    payload: dict = Depends(validate_jwt),
):
    print("hello")
    user_id = payload.get("sub", "anonymous")
    print(user_id)
    output = []
    c = connection.cursor()
    c.execute(
        f"SELECT assistant_id, assistant_name FROM assistants WHERE uuid = '{user_id}'"
    )
    # print(c.fetchall())
    results = c.fetchall()
    c.close()

    for row in results:
        output.append({"id": row[0], "name": row[1]})

    return output


# this way because a user can have multiple versions of the same channel_id (yt identifier) so the name provided to the assistant is used to differentiate
@app.post("/assistants/{assistant_name}")
def create_assistant(
    channel_id: str, assistant_name: str, payload: dict = Depends(validate_jwt)
):
    print("hello")
    user_id = payload.get("sub", "anonymous")
    print(user_id)
    print(channel_id)
    get_channel_transcript(assistant_name, channel_id, 3, user_id)
    channel_assistant = fastapi_assistant.ChannelAssistant(
        cache[user_id][assistant_name]["video_retrieval"]
    )

    cache[user_id][assistant_name]["assistant"] = channel_assistant

    # An object with the OpenAI client can't be pickled.
    # We save the assistant to be able to reload it even if the cache is cleared.
    channel_assistant.client = None
    with open(f"{user_id}_{channel_assistant.assistant.id}.pkl", "wb") as f:
        pickle.dump(channel_assistant, f, pickle.HIGHEST_PROTOCOL)

    # Save in DB
    c = connection.cursor()
    c.execute(
        f"""INSERT INTO assistants(
    assistant_id, 
    channel_id ,
    assistant_name,
    uuid ) VALUES
    ('{channel_assistant.assistant.id}', '{channel_id}', '{assistant_name}', '{user_id}');"""
    )
    c.execute(f"SELECT * FROM assistants")
    print(c.fetchall())
    c.execute(
        f"""INSERT INTO channels(
    channel_id ,
    assistant_name,
     uuid ) VALUES
    ('{channel_id}', '{assistant_name}', '{user_id}');"""
    )
    c.close()
    return channel_assistant.assistant.id


# probablemente se puede usar Depends() para que busque el assistant en el cache y lo cargue si no esta en vez de usar el decorator
@app.post("/threads/{user_id}/{assistant_name}")
@load_assistant
def create_thread(user_id: int, assistant_name: str, thread_id: str):
    # if not cache.get(user_id, {}).get(assistant_name, {}).get("assistant"):
    #     c = connection.cursor()
    #     c.execute(
    #         f"SELECT assistant_id FROM assistants WHERE user_id = {user_id} and assistant_name = '{assistant_name}'"
    #     )
    #     assistant_id = c.fetchone()[0]
    #     c.close()

    #     try:
    #         with open(f"{user_id}_{assistant_id}.pkl", "rb") as f:
    #             channel_assistant = pickle.load(f)
    #         cache[user_id][assistant_name]["assistant"] = channel_assistant
    #     except:
    #         return "Assistant not found"

    channel_assistant = cache[user_id][assistant_name]["assistant"]
    channel_assistant.create_thread(thread_id)

    c = connection.cursor()
    c.execute(
        f"""INSERT INTO threads(thread_id, assistant_id) VALUES
    ('{thread_id}', '{channel_assistant.assistant.id}');"""
    )

    channel_assistant.client = None
    with open(f"{user_id}_{channel_assistant.assistant.id}.pkl", "wb") as f:
        pickle.dump(channel_assistant, f, pickle.HIGHEST_PROTOCOL)


@app.post("/messages/{user_id}/{assistant_name}/{thread_id}")
@load_assistant
def create_message(user_id: int, assistant_name: str, thread_id: str, content: str):
    channel_assistant = cache[user_id][assistant_name]["assistant"]
    channel_assistant.create_message(thread_id, content)


@app.post("/runs/{user_id}/{assistant_name}/{thread_id}")
@load_assistant
def create_run(user_id: int, assistant_name: str, thread_id: str):
    channel_assistant = cache[user_id][assistant_name]["assistant"]
    channel_assistant.create_run(thread_id)


@app.get("/messages/{user_id}/{assistant_name}/{thread_id}")
@load_assistant_returning_object
async def get_messages(user_id: int, assistant_name: str, thread_id: str):
    output = []
    channel_assistant = cache[user_id][assistant_name]["assistant"]
    messages_list = await channel_assistant.get_clean_messages(thread_id)
    for msg in messages_list:
        output.append({"id": msg[0], "role": msg[1], "text": msg[2]})
    return output


# no enteindo por que tengo que poner esa llave al final despues de assistant name
@app.get("/threads/{assistant_name}")
def get_threads(assistant_name: str, payload: dict = Depends(validate_jwt)):
    print("hello")
    user_id = payload.get("sub", "anonymous")
    print(user_id)
    output = []
    c = connection.cursor()
    c.execute(
        f"""SELECT thread_id FROM threads 
          join assistants on threads.assistant_id = assistants.assistant_id
           WHERE assistants.assistant_name = '{assistant_name}'
            and assistants.uuid = '{user_id}'"""
    )
    # print(c.fetchall())
    results = c.fetchall()
    c.close()
    for row in results:
        output.append({"thread_id": row[0]})
    print(output)
    return output


@app.get("/print_messages/{user_id}/{assistant_name}/{thread_id}")
@load_assistant
async def print_messages(user_id: int, assistant_name: str, thread_id: str):
    channel_assistant = cache[user_id][assistant_name]["assistant"]
    channel_assistant.print_messages(thread_id)


# insert user into the db
@app.post("/users")
async def create_user(user: User):
    db_user = await utils.get_user_by_email(user.email)
    print(db_user)
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")

    return await utils.create_user(user)
