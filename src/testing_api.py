# curl -X POST http://127.0.0.1:8000/assistants/franco?channel_id=UCIEv3lZ_tNXHzL3ox-_uUGQ


# curl -X POST http://127.0.0.1:8000/threads/franco?channel_id=UCIEv3lZ_tNXHzL3ox-_uUGQ&thread_id=prueba

# uvicorn fastapi_main:app --reload

import requests

my_url = "http://127.0.0.1:8000"
user_id = 1
# channel_id = "UCIEv3lZ_tNXHzL3ox-_uUGQ"
channel_id = "UCmqincDKps3syxvD4hbODSg"
assistant_name = "marzo"
thread_id = "prueba_deco"

# requests.get(f"{my_url}/transcripts/{user_id}?channel_id={channel_id}&max_results=0")

requests.get(
    f"{my_url}/transcripts/{user_id}/{assistant_name}?channel_id={channel_id}&max_results=0"
)


requests.get(f"{my_url}/assistants/1")
# json_assistant = {"channel_id": channel_id, "channel_name": "prueba"}
# response = requests.post(f"{my_url}/assistants/{user_id}", params=json_assistant)
# requests.post(f"{my_url}/assistants/{user_id}?channel_id={channel_id}")

json_assistant = {"channel_id": channel_id}
response = requests.post(
    f"{my_url}/assistants/{user_id}/{assistant_name}", params=json_assistant
)


# json_thread = {"channel_id": channel_id, "thread_id": "prueba2"}
# response = requests.post(f"{my_url}/threads/{user_id}", params=json_thread)

json_thread = {"thread_id": thread_id}
response = requests.post(
    f"{my_url}/threads/{user_id}/{assistant_name}", params=json_thread
)


# json_message = {
#     "channel_id": channel_id,
#     "thread_id": "prueba2",
#     "content": "List the top 3 most interesting subjects the videos talk about",
# }
# response = requests.post(f"{my_url}/messages/{user_id}", params=json_message)

json_message = {
    "content": "List the top 3 most interesting subjects the videos talk about",
}
response = requests.post(
    f"{my_url}/messages/{user_id}/{assistant_name}/{thread_id}", params=json_message
)


# json_run = {"channel_id": channel_id, "thread_id": "prueba2"}
# response = requests.post(f"{my_url}/runs/{user_id}", params=json_run)

response = requests.post(f"{my_url}/runs/{user_id}/{assistant_name}/{thread_id}")


# response = requests.get(
#     f"{my_url}/messages/{user_id}?channel_id={channel_id}&thread_id=prueba2"
# )
# response
response = requests.get(f"{my_url}/messages/{user_id}/{assistant_name}/{thread_id}")
response


# response = requests.get(
#     f"{my_url}/print_messages/{user_id}?channel_id={channel_id}&thread_id=prueba2"
# )
# response


response = requests.get(
    f"{my_url}/print_messages/{user_id}/{assistant_name}/{thread_id}"
)
response


json_message = {
    "channel_id": channel_id,
    "thread_id": "prueba2",
    "content": "Write a blog post of 500 words about the second topic you mentioned in the previous answer. Make it engaging and useful to bring traffic to the blog.",
}
response = requests.post(f"{my_url}/messages/{user_id}", params=json_message)


json_message = {
    "channel_id": channel_id,
    "thread_id": "prueba2",
    "content": "make it longer and more technical with an example regarding sports analytics. Avoid the funny tone, engaging but not for childs",
}
response = requests.post(f"{my_url}/messages/{user_id}", params=json_message)


json_message = {
    "channel_id": channel_id,
    "thread_id": "prueba2",
    "content": "Now create a blog post with the same format characterstics as the previous one but about how to create a basketball short chart ",
}
response = requests.post(f"{my_url}/messages/{user_id}", params=json_message)


json_message = {
    "channel_id": channel_id,
    "thread_id": "prueba2",
    "content": "Can you cite the source of the data you used in the previous answer? I only see videos for shot charts in basketball using R. Was your answer from the file provided?",
}
response = requests.post(f"{my_url}/messages/{user_id}", params=json_message)


json_message = {
    "channel_id": channel_id,
    "thread_id": "prueba2",
    "content": "Can you create a blog post about shot charts in basketball based on the provided knowledge in the file? You have content about it. If you can't, just say you can't and say why but don't use general knowledge",
}
response = requests.post(f"{my_url}/messages/{user_id}", params=json_message)


json_message = {
    "channel_id": channel_id,
    "thread_id": "prueba2",
    "content": "How is that you can cite the source but you don't have access to it? Don't be lazy, write the blog post. You have the knowledge, you have the data, you have the source. You can do it. But if you won't do it, just say you can't and say why but don't use general knowledge",
}
response = requests.post(f"{my_url}/messages/{user_id}", params=json_message)


json_user = {"user": "franco2", "email": "asd"}
response = requests.post(f"{my_url}/users", params=json_user)


####
import sys
import os
import pickle

cache = {}
cache[1] = {}
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from decorators import load_assistant

os.chdir("src")

from openai import OpenAI
import pickle
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API setup
OPENAI_KEY = os.getenv("OPENAI_API_KEY")

with open(f"1_asst_h7QDDr9bQfjoz8CPgvICxCrw.pkl", "rb") as f:
    channel_assistant = pickle.load(f)

channel_assistant.client = OpenAI(api_key=OPENAI_KEY)


channel_assistant.create_message(
    "prueba_deco", "List the top 3 most interesting subjects the videos talk about"
)

channel_assistant.create_run("prueba_deco")


# channel_assistant.get_messages("prueba_deco")


channel_assistant.get_clean_messages("prueba_deco")

channel_assistant.print_messages("prueba_deco")
