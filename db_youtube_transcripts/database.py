import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

### YOUTUBE TRANSCRIPTS ###


def get_connection_youtube_transcripts():
    conn = psycopg2.connect(
        database=os.getenv("DB_NAME_YOUTUBE_TRANSCRIPTS"),
        host=os.getenv("DB_HOST_YOUTUBE_TRANSCRIPTS"),
        user=os.getenv("DB_USERNAME_YOUTUBE_TRANSCRIPTS"),
        password=os.getenv("DB_PASSWORD_YOUTUBE_TRANSCRIPTS"),
        port=os.getenv("DB_PORT_YOUTUBE_TRANSCRIPTS"),
    )
    conn.autocommit = True
    return conn


def get_db_youtube_transcripts():
    conn = get_connection_youtube_transcripts()
    try:
        yield conn
    finally:
        conn.close()
