from typing import Union
import fastapi_retrieve
import fastapi_assistant
from fastapi import FastAPI, Body, Form

app = FastAPI()

db = {}
db["franco"] = {}


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/transcripts/{user}")
def get_channel_transcript(user: str, channel_id: str, max_results: int):
    video_retrieval = fastapi_retrieve.VideoRetrieval(channel_id, max_results)
    video_retrieval.get_video_ids()
    video_retrieval.get_transcripts()
    db[user][channel_id] = {}
    db[user][channel_id]["video_retrieval"] = video_retrieval
    print(video_retrieval.all_transcripts[0:10])


@app.post("/assistants/{user}")
def create_assistant(user: str, channel_id: str):
    channel_assistant = fastapi_assistant.ChannelAssistant(
        db[user][channel_id]["video_retrieval"]
    )
    db[user][channel_id]["assistant"] = channel_assistant
    return channel_assistant.assistant.id


@app.post("/threads/{user}")
def create_thread(user: str, channel_id: str, thread_id: str):
    channel_assistant = db[user][channel_id]["assistant"]
    channel_assistant.create_thread(thread_id)


@app.post("/messages/{user}")
def create_message(user: str, channel_id: str, thread_id: str, content: str):
    channel_assistant = db[user][channel_id]["assistant"]
    channel_assistant.create_message(thread_id, content)


@app.post("/runs/{user}")
def create_run(user: str, channel_id: str, thread_id: str):
    channel_assistant = db[user][channel_id]["assistant"]
    channel_assistant.create_run(thread_id)


@app.get("/messages/{user}")
def get_messages(user: str, channel_id: str, thread_id: str):
    channel_assistant = db[user][channel_id]["assistant"]
    return channel_assistant.get_messages(thread_id)


@app.get("/print_messages/{user}")
def print_messages(user: str, channel_id: str, thread_id: str):
    channel_assistant = db[user][channel_id]["assistant"]
    channel_assistant.print_messages(thread_id)
