from dotenv import load_dotenv

load_dotenv()
import os

global connection

import psycopg2

connection = psycopg2.connect(database=os.getenv("DB_NAME"),
                        host=os.getenv("DB_HOST"),
                        user=os.getenv("DB_USERNAME"),
                        password=os.getenv("DB_PASSWORD"),
                        port=os.getenv("DB_PORT")
                          )

connection.autocommit = True

global cache
cache = {}
# 1 is user_id: 1
cache[1] = {}

c = connection.cursor()
c.execute("SELECT * FROM users")

c.fetchall()
# c.close()


# c = connection.cursor()
# c.execute("SELECT * FROM assistants")

# # c.fetchall()
# a = c.fetchall()
# a
# c.close()


# c = connection.cursor()
# c.execute("SELECT * FROM channels")
# c.fetchall()
# c=connection.cursor()

# c.execute(f"""
# CREATE TABLE user (
#      id MEDIUMINT NOT NULL AUTO_INCREMENT,
#      name CHAR(100) NOT NULL,
#      email CHAR(100) NOT NULL,
#      PRIMARY KEY (id)
# );

# """)


# c.execute(f"""
# DROP TABLE IF EXISTS user_delete;
# CREATE TABLE user_delete (
#      id MEDIUMINT NOT NULL AUTO_INCREMENT,
#      name VARCHAR(100) NOT NULL,
#      email VARCHAR(100) NOT NULL,
#      PRIMARY KEY (id)
# );

# INSERT INTO user_delete (name, email) VALUES
#     ('franco', 'f@g.com');

# """)

# c.close()


# c = connection.cursor()
# c.execute(f"""select * from users where email = '{email}';""")
# user = c.fetchone()
# c.close()