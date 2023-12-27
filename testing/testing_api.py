# curl -X POST http://127.0.0.1:8000/assistants/franco?channel_id=UCIEv3lZ_tNXHzL3ox-_uUGQ


# curl -X POST http://127.0.0.1:8000/threads/franco?channel_id=UCIEv3lZ_tNXHzL3ox-_uUGQ&thread_id=prueba


import requests

my_url = "http://127.0.0.1:8000"
user = "franco"
channel_id = "UCIEv3lZ_tNXHzL3ox-_uUGQ"


requests.get(f"{my_url}/transcripts/{user}?channel_id={channel_id}&max_results=2")


requests.post(f"{my_url}/assistants/{user}?channel_id={channel_id}")


json_thread = {"channel_id": channel_id, "thread_id": "prueba"}
response = requests.post(f"{my_url}/threads/{user}", params=json_thread)


json_message = {
    "channel_id": channel_id,
    "thread_id": "prueba",
    "content": " What recipes do you have?",
}
response = requests.post(f"{my_url}/messages/{user}", params=json_message)

json_run = {"channel_id": channel_id, "thread_id": "prueba"}
response = requests.post(f"{my_url}/runs/{user}", params=json_run)

response = requests.get(
    f"{my_url}/messages/{user}?channel_id={channel_id}&thread_id=prueba"
)
response
# requests.post(url,json={your data dict})


response = requests.get(
    f"{my_url}/print_messages/{user}?channel_id={channel_id}&thread_id=prueba"
)
response
