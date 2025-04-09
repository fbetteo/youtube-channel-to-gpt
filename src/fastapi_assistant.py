import os
import json
from dotenv import load_dotenv
from openai import OpenAI
import pickle
import io
import re

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
        self.file_metadata = {}  # Map file IDs to video metadata

        # Upload all transcript files
        self.file_ids = []
        for transcript_path in self.video_retrieval.transcript_files:
            with open(transcript_path, "rb") as f:
                file = self.client.files.create(file=f, purpose="assistants")
                self.file_ids.append(file.id)

                # Extract video ID from the file name
                try:
                    video_id = (
                        os.path.basename(transcript_path)
                        .split("_")[1:]
                        .join("_")
                        .replace(".txt", "")
                    )
                    if video_id in self.video_retrieval.video_metadata:
                        self.file_metadata[file.id] = (
                            self.video_retrieval.video_metadata[video_id]
                        )
                    else:
                        print(f"Warning: Video ID '{video_id}' not found in metadata.")
                except IndexError:
                    print(
                        f"Error: Could not extract video ID from file name '{transcript_path}'."
                    )

        self.assistant = self.client.beta.assistants.create(
            name="FastAPI V2 test",
            instructions=(
                """You are assisting users with information from multiple YouTube videos whose transcripts are uploaded in your knowledge files.
                When you provide sources, reference the video title or a direct link instead 
                of chunk indexes. For example: [source: 'MyVideoTitle' - https://youtu.be/<id>]. 
                Constraints
                1: Never give an answer that has not been checked in the uploaded files.
                2. You can only answer questions that can be found in the provided data sources. If you cannot find an answer,  say you don't know."""
            ),
            model="gpt-4o-mini-2024-07-18",
            tools=[{"type": "file_search"}],
            temperature=0.01,
        )

        vector_store = self.client.vector_stores.create(
            name="FastAPI V3 test", file_ids=self.file_ids
        )

        self.assistant = self.client.beta.assistants.update(
            assistant_id=self.assistant.id,
            tool_resources={"file_search": {"vector_store_ids": [vector_store.id]}},
        )

    def create_thread(self, thread_id: str):
        if not self.client:
            self.client = OpenAI(api_key=OPENAI_KEY)
        self.threads[thread_id] = self.client.beta.threads.create()
        # return self.thread

    # Optional: post-processing to replace chunk references.
    def refine_sources_in_response(self, content: str, annotations: list) -> str:
        """
        Replace source references in the content with video metadata (title and link) based on file IDs in annotations.
        """
        for annotation in annotations:
            if annotation.type == "file_citation":
                file_id = annotation.file_citation.file_id
                source_text = annotation.text
                metadata = self.file_metadata.get(file_id)

                if metadata:
                    # Replace the source text with the video title and link
                    replacement = f"[source: '{metadata}]"
                    content = content.replace(source_text, replacement)
                else:
                    # If metadata is missing, replace with a placeholder
                    content = content.replace(source_text, "[source: Unknown]")

        return content

    def create_message(self, thread_id: str, content: str):
        if not self.client:
            self.client = OpenAI(api_key=OPENAI_KEY)
        try:
            # Optionally refine old references before sending them
            content = self.refine_sources_in_response(content)
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
            # Post-process the assistant’s raw text before returning
            content = msg.content[0].text.value
            annotations = msg.content[0].text.annotations

            # Refine the content by replacing source references
            new_content = self.refine_sources_in_response(content, annotations)
            clean_messages.append((i, msg.role, new_content))

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

    def delete_files(self):
        for file_id in self.file_ids:
            file_deletion_status = self.client.beta.assistants.files.delete(
                assistant_id=self.assistant.id, file_id=file_id
            )
            print(file_deletion_status)
