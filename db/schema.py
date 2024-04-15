from dotenv import load_dotenv
import os

load_dotenv()
import psycopg2

connection = psycopg2.connect(
    database=os.getenv("DB_NAME"),
    host=os.getenv("DB_HOST"),
    user=os.getenv("DB_USERNAME"),
    password=os.getenv("DB_PASSWORD"),
    port=os.getenv("DB_PORT"),
)

connection.autocommit = True


c = connection.cursor()
c.execute(
    """
DROP TABLE IF EXISTS users;
CREATE TABLE users (
    user_id SERIAL PRIMARY KEY,
    email VARCHAR(100) NOT NULL,
    hashed_password VARCHAR(100) NOT NULL
);
"""
)

# channel name unique since the user can have multiple versions of the same channel?
# channel id is the Youtube identifier
c.execute(
    """
DROP TABLE IF EXISTS channels;
CREATE TABLE channels (
    user_id INTEGER NOT NULL,
    channel_id VARCHAR(100) NOT NULL,
    assistant_name VARCHAR(100) NOT NULL,
    PRIMARY KEY (user_id, assistant_name)
);
"""
)


# c.execute(
#     f"""
#     DROP TABLE IF EXISTS assistants;
# CREATE TABLE assistants (
#     user_id MEDIUMINT NOT NULL,
#     assistant_id CHAR(100) NOT NULL,
#     channel_id CHAR(100) NOT NULL,
#     assistant_name CHAR(100) NOT NULL,
#     PRIMARY KEY (assistant_id)
# );

# """
# )


c.execute(
    """
DROP TABLE IF EXISTS assistants;
CREATE TABLE assistants (
    user_id INTEGER NOT NULL,
    assistant_id VARCHAR(100) NOT NULL,
    channel_id VARCHAR(100) NOT NULL,
    assistant_name VARCHAR(100) NOT NULL,
    PRIMARY KEY (assistant_id)
);
"""
)


c.execute(
    """
DROP TABLE IF EXISTS threads;
CREATE TABLE threads (
    assistant_id VARCHAR(100) NOT NULL,
    thread_id VARCHAR(100) NOT NULL,
    thread_name VARCHAR(100) NOT NULL,
    PRIMARY KEY (thread_id)
    );
"""
)


c.close()
