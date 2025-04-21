import sys
import os
import logging  # use logging for production messages

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import pickle
from typing import List, Union
import fastapi_retrieve
import fastapi_assistant
from fastapi import (
    FastAPI,
    Body,
    Form,
    HTTPException,
    Request,
    status,
    Depends,
    Header,
    Security,
    Cookie,
)
from fastapi.middleware.cors import CORSMiddleware
from db.database import get_db, get_cache
from db.models import User
import json
from decorators import load_assistant, load_assistant_returning_object
import utils
import uuid
import time
from openai import OpenAI
from stripe import StripeError, stripe
from stripe.api_resources import event as stripe_event

from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from typing import Optional
from supabase import create_client, Client

app = FastAPI()

url: str = os.getenv("NEXT_PUBLIC_SUPABASE_URL")
anon_key: str = os.getenv("SUPABASE_ANON_KEY")
secret_key: str = os.getenv("SUPABASE_SECRET")

FRONTEND_URL = os.getenv("FRONTEND_URL")


supabase: Client = create_client(url, anon_key)

STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY_TEST")
OPENAI_KEY = os.getenv("OPENAI_API_KEY")

ALGORITHM = "HS256"

client = OpenAI(api_key=OPENAI_KEY)

stripe.api_key = STRIPE_SECRET_KEY

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
    FRONTEND_URL,
    "http://localhost:3000",
    "https://youtube-channel-to-gpt-frontend.vercel.app",
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
    assistant_name: str,
    channel_name: str,
    max_results: int,
    user_id: str,
    cache=Depends(get_cache),
):

    print("hello")
    print(user_id)
    video_retrieval = fastapi_retrieve.VideoRetrieval(channel_name, max_results)
    try:
        video_retrieval.get_channel_id()
    except Exception as e:
        raise HTTPException(
            status_code=400, detail="Error in get_channel_id()" + str(e)
        )

    try:
        video_retrieval.get_video_ids()
    except Exception as e:
        raise HTTPException(status_code=400, detail="Error in get_video_ids()" + str(e))

    try:
        video_retrieval.get_transcripts()
    except Exception as e:
        raise HTTPException(
            status_code=400, detail="Error in get_transcripts()" + str(e)
        )
    if cache.get(user_id) is None:
        cache[user_id] = {}
    cache[user_id][assistant_name] = {}
    cache[user_id][assistant_name]["video_retrieval"] = video_retrieval
    print(video_retrieval.all_transcripts[0:10])
    return {
        "message": "Transcripts loaded",
        "sample": video_retrieval.all_transcripts[0:10],
    }


@app.get("/assistants/{user_id}")
def get_assistants(user_id: str, user_uuid: str, db=Depends(get_db)):
    output = []
    c = db.cursor()
    c.execute(
        "SELECT assistant_id, assistant_name FROM assistants WHERE user_id = %s and uuid = %s",
        (user_id, user_uuid),
    )
    results = c.fetchall()
    c.close()

    for row in results:
        output.append({"id": row[0], "name": row[1]})

    return output


@app.get("/assistants-protected")
def get_assistants_protected(payload: dict = Depends(validate_jwt), db=Depends(get_db)):
    user_id = payload.get("sub", "anonymous")
    logging.info("User %s requested protected assistants", user_id)
    output = []
    c = db.cursor()
    c.execute(
        "SELECT assistant_id, assistant_name FROM assistants WHERE uuid = %s",
        (user_id,),
    )
    # print(c.fetchall())
    results = c.fetchall()
    c.close()

    for row in results:
        output.append({"id": row[0], "name": row[1]})

    return output


# this way because a user can have multiple versions of the same channel_id (yt identifier) so the name provided to the assistant is used to differentiate
@app.post("/assistants/{assistant_name}")
async def create_assistant(
    request: Request,
    assistant_name: str,
    payload: dict = Depends(validate_jwt),
    db=Depends(get_db),
    cache=Depends(get_cache),
):
    body = await request.json()
    channel_name = body.get("channel_name")
    print("hello")
    user_id = payload.get("sub", "anonymous")
    print(user_id)
    print(channel_name)
    get_channel_transcript(assistant_name, channel_name, 5, user_id, cache)
    channel_assistant = fastapi_assistant.ChannelAssistant(
        cache[user_id][assistant_name]["video_retrieval"]
    )
    channel_id = cache[user_id][assistant_name]["video_retrieval"].channel_id

    cache[user_id][assistant_name]["assistant"] = channel_assistant

    # An object with the OpenAI client can't be pickled.
    # We save the assistant to be able to reload it even if the cache is cleared.
    channel_assistant.client = None
    # with open(f"{user_id}_{channel_assistant.assistant.id}.pkl", "wb") as f:
    #     pickle.dump(channel_assistant, f, pickle.HIGHEST_PROTOCOL)

    # Save in DB
    c = db.cursor()
    c.execute(
        """INSERT INTO assistants(assistant_id, channel_name, channel_id, assistant_name, uuid)
           VALUES (%s, %s, %s, %s, %s)""",
        (
            channel_assistant.assistant.id,
            channel_name,
            channel_id,
            assistant_name,
            user_id,
        ),
    )
    c.execute("SELECT * FROM assistants")
    print(c.fetchall())
    c.execute(
        """INSERT INTO channels(channel_name, channel_id, assistant_name, uuid)
           VALUES (%s, %s, %s, %s)""",
        (channel_name, channel_id, assistant_name, user_id),
    )
    c.close()
    return channel_assistant.assistant.id


# probablemente se puede usar Depends() para que busque el assistant en el cache y lo cargue si no esta en vez de usar el decorator
@app.post("/threads/{assistant_name}")
# @load_assistant
def create_thread(
    assistant_name: str, payload: dict = Depends(validate_jwt), db=Depends(get_db)
):

    user_id = payload.get("sub", "anonymous")
    print(user_id)
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

    # channel_assistant = cache[user_id][assistant_name]["assistant"]

    c = db.cursor()
    c.execute(
        """SELECT max(thread_name), max(assistants.assistant_id)
           FROM threads JOIN assistants ON threads.assistant_id = assistants.assistant_id
           WHERE assistants.assistant_name = %s AND assistants.uuid = %s
           AND left(thread_name,8) = 'untitled'""",
        (assistant_name, user_id),
    )

    results = c.fetchall()
    c.close()

    if results[0][0] is None:
        thread_name = "untitled1"

        c = db.cursor()
        c.execute(
            """SELECT max(assistants.assistant_id) FROM assistants 
           WHERE assistants.assistant_name = %s AND assistants.uuid = %s""",
            (assistant_name, user_id),
        )

        results = c.fetchall()
        c.close()
        # if there are no threads, we need to get the assistant id name
        assistant_id = results[0][0]
    else:
        thread_name = f"untitled{int(results[0][0][8:]) + 1}"
        assistant_id = results[0][1]

    print(thread_name)

    new_thread = client.beta.threads.create()
    print(new_thread.id)
    # channel_assistant.create_thread(thread_id)

    c = db.cursor()
    c.execute(
        "INSERT INTO threads(thread_id, thread_name, assistant_id) VALUES (%s, %s, %s)",
        (new_thread.id, thread_name, assistant_id),
    )
    c.close()

    return {"thread_id": new_thread.id, "thread_name": thread_name}
    # channel_assistant.client = None
    # with open(f"{user_id}_{channel_assistant.assistant.id}.pkl", "wb") as f:
    #     pickle.dump(channel_assistant, f, pickle.HIGHEST_PROTOCOL)


@app.post("/messages/{assistant_id}/{thread_id}")
async def create_message(
    assistant_id: str,
    thread_id: str,
    request: Request,
    payload: dict = Depends(validate_jwt),
    db=Depends(get_db),
):
    user_id = payload.get("sub", "anonymous")
    print(user_id)
    body = await request.json()
    content = body.get("content")

    try:
        # Attempt to send the message
        message = client.beta.threads.messages.create(
            thread_id=thread_id, role="user", content=content
        )

        # Increment the user's message count and decrease remaining messages only if the message is successfully sent
        c = db.cursor()
        c.execute(
            f"""UPDATE users 
            SET count_messages = count_messages + 1, 
                remaining_messages = remaining_messages - 1 
            WHERE uuid = '{user_id}';"""
        )
        c.close()

        run = client.beta.threads.runs.create_and_poll(
            thread_id=thread_id,
            assistant_id=assistant_id,
            max_completion_tokens=1000,
            additional_instructions="Answer the question as best as you can but don't exceed 750 words approximately. This doesn't mean you have to write 750 words, but if you can answer the question in 100 words, do it. If you need to write 2000 words, cap it to 750 approx.",
        )

        if run.status == "completed":
            messages = client.beta.threads.messages.list(thread_id=thread_id)
            clean_messages = []
            for i, msg in enumerate(messages.data[::-1]):
                content = msg.content[0].text.value
                annotations = msg.content[0].text.annotations

                # Refine the content by replacing source references
                new_content = refine_sources_in_response(content, annotations)
                clean_messages.append({"id": i, "role": msg.role, "text": new_content})
            print(clean_messages)
            return clean_messages
        else:
            print(run.status)
    except Exception as e:
        print(f"Error sending message: {e}")
        raise HTTPException(status_code=500, detail="Failed to send message")


@app.post("/runs/{user_id}/{assistant_name}/{thread_id}")
# @load_assistant
def create_run(
    user_id: int, assistant_name: str, thread_id: str, cache=Depends(get_cache)
):
    channel_assistant = cache[user_id][assistant_name]["assistant"]
    channel_assistant.create_run(thread_id)


@app.get("/messages/{assistant_id}/{thread_id}")
# @load_assistant_returning_object
async def get_messages(
    assistant_id: str, thread_id: str, payload: dict = Depends(validate_jwt)
):
    user_id = payload.get("sub", "anonymous")
    print(user_id)
    messages = client.beta.threads.messages.list(thread_id=thread_id)
    clean_messages = []
    for i, msg in enumerate(messages.data[::-1]):
        # clean_messages.append(
        #     {"id": i, "role": msg.role, "text": msg.content[0].text.value}
        # )
        content = msg.content[0].text.value
        annotations = msg.content[0].text.annotations

        # Refine the content by replacing source references
        new_content = refine_sources_in_response(content, annotations)
        clean_messages.append({"id": i, "role": msg.role, "text": new_content})
    print(clean_messages)
    return clean_messages
    # messages_list = await channel_assistant.get_clean_messages(thread_id)
    # for msg in messages_list:
    #     output.append({"id": msg[0], "role": msg[1], "text": msg[2]})
    # return output


# no enteindo por que tengo que poner esa llave al final despues de assistant name
@app.get("/threads/{assistant_name}")
def get_threads(
    assistant_name: str, payload: dict = Depends(validate_jwt), db=Depends(get_db)
):
    user_id = payload.get("sub", "anonymous")
    c = db.cursor()
    c.execute(
        """SELECT thread_id,  thread_name FROM threads 
          join assistants on threads.assistant_id = assistants.assistant_id
           WHERE assistants.assistant_name = %s
            and assistants.uuid = %s""",
        (assistant_name, user_id),
    )
    results = c.fetchall()
    c.close()
    output = [{"thread_id": row[0], "thread_name": row[1]} for row in results]
    return output


@app.get("/print_messages/{user_id}/{assistant_name}/{thread_id}")
# @load_assistant
async def print_messages(
    user_id: int, assistant_name: str, thread_id: str, cache=Depends(get_cache)
):
    channel_assistant = cache[user_id][assistant_name]["assistant"]
    channel_assistant.print_messages(thread_id)


# insert user into the db
# @app.post("/users")
# async def create_user(user: User):
#     db_user = await utils.get_user_by_email(user.email)
#     print(db_user)
#     if db_user:
#         raise HTTPException(status_code=400, detail="Email already registered")

#     return await utils.create_user(user)


@app.post("/users")
async def create_user_data(
    user: User, 
    # payload: dict = Depends(validate_jwt),
    db=Depends(get_db)
):
    # user_id = payload.get("sub", "anonymous")

    c = db.cursor()
    c.execute(
        "INSERT INTO users(uuid, email, subscription, remaining_messages) VALUES (%s, %s, %s, %s)",
        (user.user_id, user.email, user.subscription, user.remaining_messages),
    )
    c.close()


@app.post("/increment_user_messages")
async def increment_user_meesages(
    payload: dict = Depends(validate_jwt), db=Depends(get_db)
):
    user_id = payload.get("sub", "anonymous")
    c = db.cursor()
    c.execute(
        f"""UPDATE users SET count_messages = count_messages +1 where uuid = '{user_id}';"""
    )
    c.close()


@app.post("/modify_user_subscription")
async def modify_user_subscription(
    subscription: str, payload: dict = Depends(validate_jwt), db=Depends(get_db)
):
    user_id = payload.get("sub", "anonymous")
    c = db.cursor()
    c.execute(
        f"""UPDATE users SET subscription = '{subscription}' where uuid = '{user_id}';"""
    )
    c.close()


@app.get("/get_user_data")
async def get_user_data(payload: dict = Depends(validate_jwt), db=Depends(get_db)):
    user_id = payload.get("sub", "anonymous")
    c = db.cursor()
    c.execute(
        f"""SELECT uuid, email, subscription, count_messages, remaining_messages FROM users where uuid = '{user_id}';"""
    )
    results = c.fetchall()
    c.close()
    output = {}
    output["uuid"] = results[0][0]
    output["email"] = results[0][1]
    output["subscription"] = results[0][2]
    output["count_messages"] = results[0][3]
    output["remaining_messages"] = results[0][4]

    return output


@app.post("/create-checkout-session")
async def create_checkout_session(
    request: Request, payload: dict = Depends(validate_jwt), db=Depends(get_db)
):
    body = await request.json()
    user_uuid = body.get("user_uuid")
    print(user_uuid)

    try:
        # Create a one-time payment checkout session
        checkout_session = stripe.checkout.Session.create(
            payment_method_types=["card"],
            line_items=[{"price": "price_1RBltfCakpeOUC7BHetgN3x9", "quantity": 1}],
            mode="payment",
            success_url=FRONTEND_URL + "/success?session_id={CHECKOUT_SESSION_ID}",
            cancel_url=FRONTEND_URL + "/cancel",
            metadata={"user_uuid": user_uuid},
        )
        return {"url": checkout_session.url}
    except StripeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/cancel-subscription")
async def cancel_subscription(
    payload: dict = Depends(validate_jwt), db=Depends(get_db)
):
    user_id = payload.get("sub", "anonymous")
    c = db.cursor()
    c.execute("SELECT subscription_id FROM users WHERE uuid = %s", (user_id,))
    results = c.fetchall()
    c.close()
    subscription_id = results[0][0]
    logging.info("Cancelling subscription %s for user %s", subscription_id, user_id)
    try:
        # Cancel the subscription at period end
        subscription = stripe.Subscription.modify(
            subscription_id,  # Replace 'sub_xxx' with your actual subscription ID
            metadata={
                "user_uuid": user_id  # Pass UUID to Stripe session for later use in webhooks
            },
        )
        cancellation = stripe.Subscription.cancel(
            subscription_id,
        )
        return {"status": "success", "cancellation": cancellation}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# Stripe API setup
WEBHOOK_SECRET = os.getenv(
    "STRIPE_WEBHOOK_SECRET_TEST"
)  # Set this to your Stripe webhook secret


@app.post("/webhook")
async def stripe_webhook(request: Request, db=Depends(get_db)):
    payload = await request.body()
    # print(payload)
    sig_header = request.headers.get("stripe-signature")

    try:
        event = stripe.Webhook.construct_event(payload, sig_header, WEBHOOK_SECRET)
    except ValueError:
        # Invalid payload
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid payload"
        )
    except stripe.error.SignatureVerificationError:
        # Invalid signature
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid signature"
        )

    # Handle the event
    if event["type"] == "checkout.session.completed":
        session = event["data"]["object"]
        user_uuid = session["metadata"]["user_uuid"]

        # Increment remaining_messages by 200 for the user
        c = db.cursor()
        c.execute(
            """UPDATE users SET remaining_messages = remaining_messages + 200 WHERE uuid = %s""",
            (user_uuid,),
        )
        c.close()

    return {"status": "success"}


def update_user_subscription(
    user_id: uuid, subscription_id: str, subscription_type: str, db=Depends(get_db)
):
    logging.info("Updating user subscription for %s", user_id)
    c = db.cursor()
    c.execute(
        "UPDATE users SET subscription_id = %s, subscription = %s WHERE uuid = %s",
        (subscription_id, subscription_type, user_id),
    )
    c.close()


def handle_failed_payment(email: str):
    print("Failed payment")
    # Handle failed payment scenario
    # Log the failure, notify the user, etc.
    pass


import requests


def refine_sources_in_response(content: str, annotations: list) -> str:
    """
    Replace source references in the content with video metadata (title and link) based on file IDs in annotations.
    """
    for annotation in annotations:
        if annotation.type == "file_citation":
            file_id = annotation.file_citation.file_id
            source_text = annotation.text

            metadata = client.files.retrieve(file_id=file_id).filename.replace(
                ".txt", ""
            )
            video_name = metadata.split("_")[0]
            video_id = metadata.split("_")[1]
            video_link = f"https://youtu.be/{video_id}"

            if metadata:
                # Replace the source text with the video title and link in Markdown format
                replacement = f"\n\nSource: [{video_name}]({video_link})"
                content = content.replace(source_text, replacement)
            else:
                # If metadata is missing, replace with a placeholder
                content = content.replace(source_text, "[source: Unknown]")

    return content
