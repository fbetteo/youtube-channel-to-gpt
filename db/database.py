import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()


def get_connection():
    conn = psycopg2.connect(
        database=os.getenv("DB_NAME"),
        host=os.getenv("DB_HOST"),
        user=os.getenv("DB_USERNAME"),
        password=os.getenv("DB_PASSWORD"),
        port=os.getenv("DB_PORT"),
    )
    conn.autocommit = True
    return conn


def get_db():
    conn = get_connection()
    try:
        yield conn
    finally:
        conn.close()


# A simple shared cache object.
cache = {}


def get_cache():
    global cache
    return cache
