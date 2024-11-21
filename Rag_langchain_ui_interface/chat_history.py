import uuid
import os
import psycopg
from dotenv import load_dotenv
from langchain_postgres import PostgresChatMessageHistory
from dotenv import load_dotenv
load_dotenv()

DB_CONNECTION_URL_2 = os.getenv("DB_CONNECTION_URL_2")

def initialize_chat_history(session_id=None):
    connection = psycopg.connect(DB_CONNECTION_URL_2)
    table_name = "chat_history"
    PostgresChatMessageHistory.create_tables(connection, table_name)

    if session_id is None:
        session_id = str(uuid.uuid4())
    chat_history = PostgresChatMessageHistory(
        table_name, session_id, sync_connection=connection
    )
    return chat_history

def insert_chat_message(chat_history, query, response):
    chat_history.add_user_message(query)
    chat_history.add_ai_message(response)
