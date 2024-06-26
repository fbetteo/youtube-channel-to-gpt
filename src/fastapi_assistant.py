import os
import json
from dotenv import load_dotenv
from openai import OpenAI
import pickle
import io

# Load environment variables
load_dotenv()

# API setup
OPENAI_KEY = os.getenv("OPENAI_API_KEY")

import fastapi_retrieve


# video_retrieval = fastapi_retrieve.VideoRetrieval("UCIEv3lZ_tNXHzL3ox-_uUGQ", 2)
# video_retrieval.get_video_ids()
# video_retrieval.get_transcripts()


class ChannelAssistant:
    # https://platform.openai.com/docs/assistants/overview
    # TODO: add some identifier per user, or version, etc Name of the assistant should change to avoid duplication if the same channel is used twice.
    def __init__(self, video_retrieval: fastapi_retrieve.VideoRetrieval):
        self.video_retrieval = video_retrieval
        self.client = OpenAI(api_key=OPENAI_KEY)
        self.threads = {}

        # client = OpenAI(api_key=OPENAI_KEY)

        self.file = self.client.files.create(
            file=(
                "uploaded_file.txt",
                self.video_retrieval.all_transcripts.encode("utf-8"),
            ),
            purpose="assistants",
        )

        print(type(self.file))
        print(type(self.file.id))

        # self.assistant = self.client.beta.assistants.create(
        #     name="FastAPI test",
        #     instructions="You will answer as if you are the owner of the youtube channel where the files provided are from. The user is asking you questions about the videos. You will answer based on your knowledge of the videos and the channel. Be as helpful as possible. Be concise and to the point. If you do not know the answer, you can say 'I don't know'. Put the source of the answer. Provide lists when possible, make it easy to understand.",
        #     tools=[{"type": "retrieval"}],
        #     model="gpt-4-1106-preview",
        #     file_ids=[self.file.id],
        # )

        self.assistant = self.client.beta.assistants.create(
            name="FastAPI V2 test",
            instructions="You will answer as if you are the owner of the youtube channel where the files provided are from. The user is asking you questions about the videos. You will answer based on your knowledge of the videos and the channel. Be as helpful as possible. Be concise and to the point. If you do not know the answer, you can say 'I don't know'. Put the source of the answer. Provide lists when possible, make it easy to understand. Answers should be concise and no matter what you shouldn't answer longer phrases if the questions asks for it.",
            model="gpt-3.5-turbo-0125",
            tools=[{"type": "file_search"}],
            temperature=0.01,
        )
        # Create a vector store caled "Financial Statements"
        # vector_store = self.client.beta.vector_stores.create(name="FastAPI V2 test")

        vector_store = self.client.beta.vector_stores.create(
            name="FastAPI V2 test", file_ids=[self.file.id]
        )

        # Ready the files for upload to OpenAI
        # COMENTO PORQUE YA LO TENGO SUPONGO
        ## TENGO QUE PASARLO A BYTES
        # file_paths = ["edgar/goog-10k.pdf", "edgar/brka-10k.txt"]
        # file_streams = [
        #     io.BytesIO(self.video_retrieval.all_transcripts.encode("utf-8"))
        # ]

        # Use the upload and poll SDK helper to upload the files, add them to the vector store,
        # and poll the status of the file batch for completion.
        # file_batch = self.client.beta.vector_stores.file_batches.upload_and_poll(
        #     vector_store_id=vector_store.id,
        #     files=file_streams,
        # )

        # # You can print the status and the file counts of the batch to see the result of this operation.
        # print(file_batch.status)
        # print(file_batch.file_counts)

        self.assistant = self.client.beta.assistants.update(
            assistant_id=self.assistant.id,
            tool_resources={"file_search": {"vector_store_ids": [vector_store.id]}},
        )

    def create_thread(self, thread_id: str):
        if not self.client:
            self.client = OpenAI(api_key=OPENAI_KEY)
        self.threads[thread_id] = self.client.beta.threads.create()
        # return self.thread

    def create_message(self, thread_id: str, content: str):
        if not self.client:
            self.client = OpenAI(api_key=OPENAI_KEY)
        try:
            self.client.beta.threads.messages.create(
                thread_id=self.threads[thread_id].id, role="user", content=content
            )
        except KeyError:
            print("Thread doesn't exist")

    def create_run(self, thread_id: str):
        if not self.client:
            self.client = OpenAI(api_key=OPENAI_KEY)
        try:
            self.client.beta.threads.runs.create(
                thread_id=self.threads[thread_id].id,
                assistant_id=self.assistant.id,
                #   additional_instructions="Please address the user as Jane Doe. The user has a premium account."
            )
        except KeyError:
            print("Thread doesn't exist")  # use httpexception¿

    async def get_all_messages(self, thread_id: str):
        if not self.client:
            self.client = OpenAI(api_key=OPENAI_KEY)

        output_messages = self.client.beta.threads.messages.list(
            thread_id=self.threads[thread_id].id
        )
        return output_messages

    async def get_clean_messages(self, thread_id: str):
        messages = await self.get_all_messages(thread_id)
        clean_messages = []
        for i, msg in enumerate(messages.data[::-1]):
            clean_messages.append((i, msg.role, msg.content[0].text.value))
        return clean_messages

    # def get_threads(self):
    #     if not self.client:
    #         self.client = OpenAI(api_key=OPENAI_KEY)
    #     return self.client.beta.threads.list()

    async def print_messages(self, thread_id: str):
        messages = await self.get_all_messages(thread_id)
        for msg in messages.data:
            print(msg.role)
            print(msg.content)

    def delete_file(self):
        # Detaching the file from the assistant removes the file from the retrieval index and means you will no longer be charged for the storage of the indexed file.
        file_deletion_status = self.client.beta.assistants.files.delete(
            assistant_id=self.assistant.id, file_id=self.file.id
        )
        print(file_deletion_status)
