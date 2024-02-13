from openai import OpenAI
import json
# open json file with api key (secrets.json)
with open('secrets.json') as f:
    secrets = json.load(f)

open_ai_key = secrets['my_test_key']

#  Validar que esto funciona tambien
# import os
# import json
# from dotenv import load_dotenv
# # Load environment variables
# load_dotenv()

# # API setup
# OPENAI_KEY = os.getenv("OPENAU_API_KEY")




client = OpenAI(api_key=open_ai_key)

file = client.files.create(
  file=open("build/transcript_ramsay.txt", "rb"),
  purpose='assistants'
)

file.id


assistant = client.beta.assistants.create(
    name="API Ramsey Test",
    instructions="You will answer as if you are the owner of the youtube channel where the files provided are from. The user is asking you questions about the videos. You will answer based on your knowledge of the videos and the channel. Be as helpful as possible. Be concise and to the point. If you do not know the answer, you can say 'I don't know'. Put the source of the answer. Provide lists when possible, make it easy to understand.",
    tools=[{"type": "retrieval"}],
    model="gpt-4-1106-preview",
    file_ids=[file.id]
)



thread = client.beta.threads.create()
thread

message = client.beta.threads.messages.create(
    thread_id=thread.id,
    role="user",
    content="Who is the owner of the channel?"
)

message = client.beta.threads.messages.create(
    thread_id=thread.id,
    role="user",
    content="What recipes do you know"
)

# thread_messages = client.beta.threads.messages.list(thread_id=thread.id)
# print(thread_messages.data)


run = client.beta.threads.runs.create(
  thread_id=thread.id,
  assistant_id=assistant.id,
#   additional_instructions="Please address the user as Jane Doe. The user has a premium account."
)

run = client.beta.threads.runs.retrieve(
  thread_id=thread.id,
  run_id=run.id
)

run

messages = client.beta.threads.messages.list(
  thread_id=thread.id
)

for msg in messages.data:
    print(msg.role)
    print(msg.content)

print(messages.data[0].content[0])


print(messages.data[0].content[0].text.value)
# Detaching the file from the assistant removes the file from the retrieval index and means you will no longer be charged for the storage of the indexed file.
file_deletion_status = client.beta.assistants.files.delete(
  assistant_id=assistant.id,
  file_id=file.id
)