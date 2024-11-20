from flask import Flask, render_template, request
import uuid
import os
import psycopg
from dotenv import load_dotenv
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.schema import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_postgres.vectorstores import PGVector
from langchain_postgres import PostgresChatMessageHistory
from langchain_groq import ChatGroq
from langchain.vectorstores import VectorStore
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

load_dotenv()

app = Flask(__name__)

DB_CONNECTION_URL_2 = os.getenv("DB_CONNECTION_URL_2")
DB_HOST = os.getenv("DB_HOST")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PORT = os.getenv("DB_PORT")
DB_PASS = os.getenv("DB_PASS")


embeddings = GoogleGenerativeAIEmbeddings(
    api_key=os.getenv("GROQ_API_KEY"),
    model="models/embedding-001"  
)

insurance_loader = CSVLoader(file_path="insurance.csv")
happiness_loader = CSVLoader(file_path="world_happiness.csv")

insurance_documents = insurance_loader.load()
happiness_documents = happiness_loader.load()

combined_documents = insurance_documents + happiness_documents

def initialize_chat_history(session_id=None):
    connection = psycopg.connect(DB_CONNECTION_URL_2)
    table_name = "chat_history"
    
    PostgresChatMessageHistory.create_tables(connection, table_name)
    
    if session_id is None:
        session_id = str(uuid.uuid4())  
    
    chat_history = PostgresChatMessageHistory(
        table_name,
        session_id,
        sync_connection=connection
    )
    
    return chat_history

def insert_chat_message(chat_history, query, response):
    chat_history.add_user_message(query)
    chat_history.add_ai_message(response)

def get_chat_history(session_id):
    connection = psycopg.connect(DB_CONNECTION_URL_2)
    cursor = connection.cursor()
    cursor.execute('''
        SELECT query, response FROM chat_history WHERE session_id = %s ORDER BY id ASC
    ''', (session_id,))
    return cursor.fetchall()


def store_combined_embeddings(documents, collection_name="combined_data_collection"):
    try:
        vectorstore = PGVector.from_documents(
            documents,
            embeddings,
            collection_name=collection_name,
            postgres_table_name="combined_embeddings"
        )
        
        print("Storing CSV embeddings...")
        vectorstore.add_documents(documents)
        print("Documents successfully added to the combined_data_collection collection.")
    except Exception as e:
        print(f"Error during embedding storage: {e}")

store_combined_embeddings(combined_documents)

def get_ai_response(query):
    try:
        vectorstore = PGVector.from_documents(
            combined_documents,
            embeddings,
            collection_name="combined_data_collection",
            postgres_table_name="combined_embeddings"
        )
        
        query_embedding = embeddings.embed_query(query)
        search_results = vectorstore.similarity_search(query_embedding, k=1)
        
        most_similar_doc = search_results[0]
        return most_similar_doc.page_content
    except Exception as e:
        print(f"Error during similarity search: {e}")
        return "Sorry, I couldn't find an answer to your query."



def start_chat():
    session_id = str(uuid.uuid4())  
    chat_history = initialize_chat_history(session_id)
    
    print("Start chatting (type 'exit' to stop)...")
    
    while True:
        query = input("Enter your query: ")
        
        if query.lower() == "exit":
            print("Exiting chat...")
            break
        
        messages = [
            ("system", "You are a helpful assistant."),
            ("human", query),
        ]
        
        chatgroq_model = ChatGroq(
            model="mixtral-8x7b-32768",  
            temperature=0.0,
            max_retries=2
        )
        
        try:
            groq_response = chatgroq_model.invoke(messages)
            
            if hasattr(groq_response, 'content'):
                print("AI Response:", groq_response.content)  
            else:
                print("AI Response:", groq_response)  
            insert_chat_message(chat_history, query, groq_response.content if hasattr(groq_response, 'content') else str(groq_response))
        except Exception as e:
            print(f"Error during Groq API call: {e}")

start_chat()
