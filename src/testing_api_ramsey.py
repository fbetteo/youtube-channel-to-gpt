from openai import OpenAI
import json
import os

# open json file with api key (secrets.json)
# with open("secrets.json") as f:
#     secrets = json.load(f)

# open_ai_key = secrets["my_test_key"]
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
#  Validar que esto funciona tambien
# import os
# import json
# from dotenv import load_dotenv
# # Load environment variables
# load_dotenv()

# # API setup
# OPENAI_KEY = os.getenv("OPENAU_API_KEY")


client = OpenAI(api_key=OPENAI_KEY)

file = client.files.create(
    file=open("build/marclou_day6.txt", "rb"), purpose="assistants"
)


file.id


assistant = client.beta.assistants.create(
    name="API Marc Lou test for Source",
    instructions="""You are assisting users with information from multiple YouTube videos whose transcripts are uploaded in your knowledge files.
                When you provide sources, reference the video title or a direct link instead 
                of chunk indexes. For example: [source: 'MyVideoTitle' - https://youtu.be/<id>]. 
                Constraints
                1: Never give an answer that has not been checked in the uploaded files.
                2. You can only answer questions that can be found in the provided data sources. If you cannot find an answer,  say you don't know""",
    tools=[{"type": "file_search"}],
    model="gpt-4o-mini-2024-07-18",
    temperature=0.01,
)

# pude modificarlo desde la UI pero no se como mandarlo desde aca.
# Update 2025: no pude crearlo desde aca, lo arme manual en la UI todavia. Muy confundido porque en fastapi si funciona parece.
vector_store = client.beta.vector_stores.create(
    name="FastAPI Marclou April 2025", file_ids=[file.id]
)


assistant = client.beta.assistants.update(
    assistant_id=assistant.id,
    tool_resources={
        "file_search": {"vector_store_ids": ["vs_67f2d8ea0c90819198fc19ccb15b3681"]}
    },
)

thread = client.beta.threads.create()
thread

message = client.beta.threads.messages.create(
    thread_id=thread.id, role="user", content="What issues are faced with Stripe?"
)

message = client.beta.threads.messages.create(
    thread_id=thread.id, role="user", content="What recipes do you know"
)

# thread_messages = client.beta.threads.messages.list(thread_id=thread.id)
# print(thread_messages.data)


run = client.beta.threads.runs.create(
    thread_id=thread.id,
    assistant_id=assistant.id,
    #   additional_instructions="Please address the user as Jane Doe. The user has a premium account."
)

run = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)

run

messages = client.beta.threads.messages.list(thread_id=thread.id)

for msg in messages.data:
    print(msg.role)
    print(msg.content)

print(messages.data[0].content[0])

messages.data[0].content[0].text.annotations


messages.data[0].content[0].text.value

import re

re.findall(r"\[(\d+:\d+)†source\]", messages.data[0].content[0].text.value)

print(messages.data[0].content[0].text.value)
# Detaching the file from the assistant removes the file from the retrieval index and means you will no longer be charged for the storage of the indexed file.
file_deletion_status = client.beta.assistants.files.delete(
    assistant_id=assistant.id, file_id=file.id
)


client.files.retrieve(file_id="file-E5C9VrvEUP2XAkfzoVsHA1").filename.replace(
    ".txt", ""
)


messages = client.beta.threads.messages.list(thread_id=thread.id)


client.beta.assistants.retrieve("asst_vrgNlJI0oiFQdk8oZCjvZglF")


run = client.beta.threads.runs.create(
    thread_id=thread.id,
    assistant_id="asst_vrgNlJI0oiFQdk8oZCjvZglF",
    #   additional_instructions="Please address the user as Jane Doe. The user has a premium account."
)
