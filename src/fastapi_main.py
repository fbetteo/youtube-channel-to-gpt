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

STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY_LIVE")
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
MAX_ASSISTANTS_PER_USER = int(os.getenv("MAX_ASSISTANTS_PER_USER", 5))


ALGORITHM = "HS256"

client = OpenAI(api_key=OPENAI_KEY)

stripe.api_key = STRIPE_SECRET_KEY

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
    "youchatchannel.com",
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
    # this is the UUID of cuentatest2. Useful to use the API without logging in in Dev mode
    # if os.getenv("DEV_MODE") == "true":
    #     return {"sub": "96933f74-278f-44e5-911b-118fc234dd5f"}

    token = credentials.credentials
    # print(token)
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
    video_retrieval = fastapi_retrieve.VideoRetrieval(
        channel_name, max_results, user_id
    )
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


# --- Dependency Function for Limit Check ---
async def verify_assistant_limit(
    payload: dict = Depends(validate_jwt), db=Depends(get_db)
):
    """
    Dependency that checks if the user has reached the maximum assistant limit.
    Raises HTTPException if the limit is exceeded.
    """
    print("Verifying assistant limit for user %s", payload.get("sub", "anonymous"))
    user_id = payload.get("sub", "anonymous")
    c = db.cursor()
    try:
        c.execute("SELECT COUNT(*) FROM assistants WHERE uuid = %s", (user_id,))
        count_result = c.fetchone()
        current_assistant_count = count_result[0] if count_result else 0
    except Exception as e:
        logging.error(
            f"Database error checking assistant count for user {user_id}: {e}"
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not verify assistant count.",
        )
    finally:
        c.close()

    if current_assistant_count >= MAX_ASSISTANTS_PER_USER:
        logging.warning(
            f"User {user_id} tried to create assistant beyond limit ({MAX_ASSISTANTS_PER_USER})."
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Maximum number of assistants ({MAX_ASSISTANTS_PER_USER}) reached.",
        )


# this way because a user can have multiple versions of the same channel_id (yt identifier) so the name provided to the assistant is used to differentiate
@app.post("/assistants/{assistant_name}")
async def create_assistant(
    request: Request,
    assistant_name: str,
    # The result of the dependency isn't needed, just its execution for the check.
    _: None = Depends(verify_assistant_limit),
    payload: dict = Depends(validate_jwt),
    db=Depends(get_db),
    cache=Depends(get_cache),
):
    body = await request.json()
    channel_name = body.get("channel_name")
    if not channel_name:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="channel_name is required in the request body.",
        )
    print("hello")
    user_id = payload.get("sub", "anonymous")
    print(user_id)
    print(channel_name)
    try:
        # Note: get_channel_transcript itself uses Depends(get_cache),
        # so passing cache explicitly might be redundant if it's refactored
        # to use Depends internally. For now, keep passing it if needed.
        get_channel_transcript(assistant_name, channel_name, 30, user_id, cache)
    except HTTPException as e:
        # Re-raise the HTTPException from get_channel_transcript
        raise e
    except Exception as e:
        # Catch other potential errors during transcript retrieval
        logging.error(
            f"Error in get_channel_transcript for user {user_id}, assistant {assistant_name}: {e}"
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process channel information.",
        )

    # Check if video_retrieval was successfully created and added to cache
    # Defensive check, ideally get_channel_transcript should raise if it fails
    if not cache.get(user_id, {}).get(assistant_name, {}).get("video_retrieval"):
        logging.error(
            f"video_retrieval not found in cache for user {user_id}, assistant {assistant_name} after get_channel_transcript call."
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve channel transcripts after processing.",
        )
    # get_channel_transcript(assistant_name, channel_name, 5, user_id, cache)
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
    try:
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
        c.execute(
            """INSERT INTO channels(channel_name, channel_id, assistant_name, uuid)
               VALUES (%s, %s, %s, %s)""",
            (channel_name, channel_id, assistant_name, user_id),
        )
        logging.info(
            f"Assistant {assistant_name} ({channel_assistant.assistant.id}) created for user {user_id}"
        )
    except Exception as e:
        logging.error(
            f"Database error during assistant creation for user {user_id}, assistant {assistant_name}: {e}"
        )
        # Consider rolling back if using transactions
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to save assistant data to the database.",
        )
    finally:
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
async def create_message_and_run(  # Renamed for clarity
    assistant_id: str,
    thread_id: str,
    request: Request,
    payload: dict = Depends(validate_jwt),
    db=Depends(get_db),
):
    user_id = payload.get("sub", "anonymous")
    print(f"User {user_id} creating message in thread {thread_id}")
    body = await request.json()
    content = body.get("content")
    if not content:
        raise HTTPException(status_code=400, detail="Content cannot be empty")

    try:
        # 1. Create the message
        message = client.beta.threads.messages.create(
            thread_id=thread_id, role="user", content=content
        )
        print(f"Message {message.id} created for thread {thread_id}")

        # 2. Increment user's message count and decrease remaining (Consider doing this *after* successful run creation)
        # It might be better to decrement remaining_messages only when the run completes successfully or fails definitively.
        # For now, keeping it here as per original logic.
        c = db.cursor()
        try:
            c.execute(
                """UPDATE users
                   SET count_messages = count_messages + 1,
                       remaining_messages = remaining_messages - 1
                   WHERE uuid = %s AND remaining_messages > 0;""",  # Added check for remaining messages
                (user_id,),
            )
            if c.rowcount == 0:
                # Handle case where user has no remaining messages
                logging.warning(
                    f"User {user_id} attempted to send message with 0 remaining messages."
                )
                # You might need to delete the created OpenAI message here if you want atomicity
                # client.beta.threads.messages.delete(thread_id=thread_id, message_id=message.id) # Example
                raise HTTPException(status_code=403, detail="No remaining messages.")
            logging.info(f"Decremented message count for user {user_id}")
        except Exception as db_exc:
            logging.error(
                f"DB Error updating message count for user {user_id}: {db_exc}"
            )
            # Consider deleting the OpenAI message if the DB update fails
            raise HTTPException(
                status_code=500, detail="Failed to update user message count."
            )
        finally:
            c.close()

        # 3. Create the run (without polling)
        run = client.beta.threads.runs.create(
            thread_id=thread_id,
            assistant_id=assistant_id,
            max_completion_tokens=1000,
            additional_instructions="Answer the question as best as you can but don't exceed 750 words approximately. This doesn't mean you have to write 750 words, but if you can answer the question in 100 words, do it. If you need to write 2000 words, cap it to 750 approx.",
        )
        print(f"Run {run.id} created for thread {thread_id}, status: {run.status}")

        # 4. Return the run_id and thread_id for polling
        return {"run_id": run.id, "thread_id": thread_id, "status": run.status}

    except HTTPException as http_exc:
        # Re-raise HTTPExceptions directly
        raise http_exc
    except Exception as e:
        print(f"Error creating message or run for thread {thread_id}: {e}")
        logging.error(
            f"Error creating message or run for thread {thread_id} by user {user_id}: {e}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=500, detail="Failed to process message or start run."
        )


@app.get("/runs/{thread_id}/{run_id}")
async def get_run_status(
    thread_id: str,
    run_id: str,
    payload: dict = Depends(validate_jwt),  # Keep authentication
):
    user_id = payload.get("sub", "anonymous")  # Optional: Log which user is checking
    print(f"User {user_id} checking status for run {run_id} in thread {thread_id}")
    try:
        run = client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run_id)

        if run.status == "completed":
            # If completed, fetch ALL messages in the thread without filtering
            messages = client.beta.threads.messages.list(
                thread_id=thread_id, order="asc"
            )  # Use order="asc" to get oldest first
            clean_messages = []

            # Include ALL messages in the thread without filtering by run_id
            for i, msg in enumerate(messages.data):  # Iterate in ascending order
                content_block = msg.content[0]
                if content_block.type == "text":
                    content = content_block.text.value
                    annotations = content_block.text.annotations
                    # Refine the content by replacing source references
                    new_content = refine_sources_in_response(content, annotations)
                    clean_messages.append(
                        {"id": msg.id, "role": msg.role, "text": new_content}
                    )  # Use message ID

            print(f"Run {run_id} completed. Returning all messages in thread.")
            return {"status": run.status, "messages": clean_messages}
        elif run.status in ["failed", "cancelled", "expired"]:
            print(
                f"Run {run_id} ended with status: {run.status}. Error: {run.last_error}"
            )
            # Optionally, you could refund the message credit here if the run failed
            # Add logic to increment remaining_messages back if needed
            error_message = (
                str(run.last_error)
                if run.last_error
                else "Run failed or was cancelled."
            )
            return {"status": run.status, "error": error_message}
        else:
            # Still in progress (queued, in_progress, requires_action, cancelling)
            print(f"Run {run_id} status: {run.status}")
            return {"status": run.status}

    except Exception as e:
        print(f"Error retrieving status for run {run_id}: {e}")
        logging.error(
            f"Error retrieving status for run {run_id} by user {user_id}: {e}",
            exc_info=True,
        )
        # Don't expose detailed errors unless necessary
        raise HTTPException(status_code=500, detail="Failed to retrieve run status.")


@app.get("/messages/{assistant_id}/{thread_id}")
async def get_messages(
    assistant_id: str, thread_id: str, payload: dict = Depends(validate_jwt)
):
    user_id = payload.get("sub", "anonymous")
    print(f"User {user_id} fetching all messages for thread {thread_id}")
    try:
        # Fetch all messages for the thread, ordered chronologically
        messages = client.beta.threads.messages.list(thread_id=thread_id, order="asc")
        clean_messages = []
        for msg in messages.data:  # Iterate in ascending order
            content_block = msg.content[0]
            if content_block.type == "text":
                content = content_block.text.value
                annotations = content_block.text.annotations
                # Refine the content by replacing source references
                new_content = refine_sources_in_response(content, annotations)
                clean_messages.append(
                    {"id": msg.id, "role": msg.role, "text": new_content}
                )  # Use message ID

        print(f"Returning all messages for thread {thread_id}")
        return clean_messages
    except Exception as e:
        print(f"Error retrieving messages for thread {thread_id}: {e}")
        logging.error(
            f"Error retrieving messages for thread {thread_id} by user {user_id}: {e}",
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail="Failed to retrieve messages.")


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
    db=Depends(get_db),
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
            line_items=[{"price": "price_1RBlhvCakpeOUC7BHkbUVA0a", "quantity": 1}],
            mode="payment",
            success_url=FRONTEND_URL + "/success?session_id={CHECKOUT_SESSION_ID}",
            cancel_url=FRONTEND_URL + "/cancel",
            metadata={"user_uuid": user_uuid, "project": "youchatchannel"},
        )
        return {"url": checkout_session.url}
    except StripeError as e:
        raise HTTPException(status_code=400, detail=str(e))


# Not used. No subscriptionw now. Just in case for boilerplate for the future
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
                "user_uuid": user_id,  # Pass UUID to Stripe session for later use in webhooks
                "project": "youchatchannel",  # Pass UUID to Stripe session for later use in webhooks
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
    "STRIPE_WEBHOOK_SECRET_LIVE"
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
        # --- Added check for project metadata ---
        metadata = session.get("metadata", {})
        user_uuid = metadata.get("user_uuid")
        project = metadata.get("project")

        if project == "youchatchannel" and user_uuid:
            logging.info(
                f"Processing checkout.session.completed for user {user_uuid} from project {project}"
            )
            # Increment remaining_messages by 200 for the user
            c = db.cursor()
            try:
                c.execute(
                    """UPDATE users SET remaining_messages = remaining_messages + 200 WHERE uuid = %s""",
                    (user_uuid,),
                )
                logging.info(
                    f"Successfully updated remaining messages for user {user_uuid}"
                )
            except Exception as e:
                logging.error(
                    f"Database error updating messages for user {user_uuid}: {e}"
                )
                # Optionally raise an internal server error or handle differently
            finally:
                c.close()
        elif not user_uuid:
            logging.warning(
                f"Received checkout.session.completed event without user_uuid in metadata. Session ID: {session.get('id')}"
            )
        else:
            logging.warning(
                f"Received checkout.session.completed event for project '{project}' (expected 'youchatchannel'). Ignoring. Session ID: {session.get('id')}"
            )

    return {"status": "success"}


# Not used. No subscriptionw now. Just in case for boilerplate for the future
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
