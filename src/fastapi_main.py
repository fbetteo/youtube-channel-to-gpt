import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import pickle
from typing import Union
import fastapi_retrieve
import fastapi_assistant
from fastapi import FastAPI, Body, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from db.test_ps import connection, cache
from db.models import User
import json
from decorators import load_assistant
import utils

app = FastAPI()


# I think this was needed to allow the frontend to connect to the backend
origins = ["http://localhost:3000", "http://127.0.0.1:3000"]

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
# cache[1] = {}


@app.get("/")
def read_root():
    return {"Hello": "World"}


# Deberiamos tener una sola funcion cuando el usuario quiera generar un assistant, que baje el transcript y cree el assistant. Asi no guardamos en ningun momento el transcript?
@app.get("/transcripts/{user_id}/{assistant_name}")
def get_channel_transcript(
    user_id: int, assistant_name: str, channel_id: str, max_results: int
):
    video_retrieval = fastapi_retrieve.VideoRetrieval(channel_id, max_results)
    video_retrieval.get_video_ids()
    video_retrieval.get_transcripts()
    cache[user_id][assistant_name] = {}
    cache[user_id][assistant_name]["video_retrieval"] = video_retrieval
    print(video_retrieval.all_transcripts[0:10])


@app.get("/assistants/{user_id}")
def get_assistants(user_id: int):
    output = []
    c = connection.cursor()
    c.execute(
        f"SELECT assistant_id, assistant_name FROM assistants WHERE user_id = {user_id}"
    )
    # print(c.fetchall())
    results = c.fetchall()
    c.close()

    for row in results:
        output.append({"id": row[0], "name": row[1]})

    return output


# this way because a user can have multiple versions of the same channel_id (yt identifier) so the name provided to the assistant is used to differentiate
@app.post("/assistants/{user_id}/{assistant_name}")
def create_assistant(user_id: int, channel_id: str, assistant_name: str):
    print(channel_id)
    get_channel_transcript(user_id, assistant_name, channel_id, 3)
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
        f"""INSERT INTO assistants(user_id ,
    assistant_id, 
    channel_id ,
    assistant_name ) VALUES
    ({user_id}, '{channel_assistant.assistant.id}', '{channel_id}', '{assistant_name}');"""
    )
    c.execute(f"SELECT * FROM assistants")
    print(c.fetchall())
    c.execute(
        f"""INSERT INTO channels(user_id ,
    channel_id ,
    assistant_name ) VALUES
    ({user_id},  '{channel_id}', '{assistant_name}');"""
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
@load_assistant
def get_messages(user_id: int, assistant_name: str, thread_id: str):
    channel_assistant = cache[user_id][assistant_name]["assistant"]
    return channel_assistant.get_messages(thread_id)


# no enteindo por que tengo que poner esa llave al final despues de assistant name
@app.get("/threads/{user_id}/{assistant_name}")
def get_threads(user_id: int, assistant_name: str):
    output = []
    c = connection.cursor()
    c.execute(
        f"""SELECT thread_id FROM threads 
          join assistants on threads.assistant_id = assistants.assistant_id
           WHERE assistants.assistant_name = '{assistant_name}'
            and assistants.user_id = {user_id}"""
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
def print_messages(user_id: int, assistant_name: str, thread_id: str):
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
