from dotenv import load_dotenv

load_dotenv()
import os
import MySQLdb

connection = MySQLdb.connect(
    host=os.getenv("DB_HOST"),
    user=os.getenv("DB_USERNAME"),
    passwd=os.getenv("DB_PASSWORD"),
    db=os.getenv("DB_NAME"),
    autocommit=True,
    ssl_mode="VERIFY_IDENTITY",
)


c = connection.cursor()

c.execute(
    f"""
 DROP TABLE IF EXISTS users;
CREATE TABLE users (
    user_id MEDIUMINT NOT NULL AUTO_INCREMENT,
    email CHAR(100) NOT NULL,
    hashed_password CHAR(100) NOT NULL,
    PRIMARY KEY (user_id)
);

"""
)

# channel name unique since the user can have multiple versions of the same channel?
# channel id is the Youtube identifier
c.execute(
    f"""
    DROP TABLE IF EXISTS channels;
CREATE TABLE channels (
    user_id MEDIUMINT NOT NULL,
    channel_id CHAR(100) NOT NULL,
    assistant_name CHAR(100) NOT NULL,
    PRIMARY KEY (user_id, assistant_name)
);

"""
)

c.execute(
    f"""
    DROP TABLE IF EXISTS assistants;
CREATE TABLE assistants (
    user_id MEDIUMINT NOT NULL,
    assistant_id CHAR(100) NOT NULL,
    channel_id CHAR(100) NOT NULL,
    assistant_name CHAR(100) NOT NULL,
    PRIMARY KEY (assistant_id)
);

"""
)


c.execute(
    f"""
    DROP TABLE IF EXISTS threads;
CREATE TABLE threads (
    assistant_id CHAR(100) NOT NULL,
    thread_id CHAR(100) NOT NULL,
    PRIMARY KEY (thread_id)
);
"""
)

c.close()
