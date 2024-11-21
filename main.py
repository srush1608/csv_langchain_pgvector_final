
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
from langchain_core.prompts import PromptTemplate

load_dotenv()

app = Flask(__name__)

DB_CONNECTION_URL_2 = os.getenv("DB_CONNECTION_URL_2")
DB_HOST = os.getenv("DB_HOST")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PORT = os.getenv("DB_PORT")
DB_PASS = os.getenv("DB_PASS")

embeddings = GoogleGenerativeAIEmbeddings(
    api_key=os.getenv("GOOGLE_API_KEY"),
    model="models/embedding-001"
)

chatgroq_model = ChatGroq(
    model="mixtral-8x7b-32768",
    temperature=0.0,
    max_retries=2
)

# Check if embeddings already exist in the database
def embeddings_exist(vectorstore_table_name="combined_embeddings"):
    try:
        # Try to connect and check if the table exists
        connection = psycopg.connect(DB_CONNECTION_URL_2)
        cursor = connection.cursor()
        
        # Check if the table exists
        cursor.execute(f"""
            SELECT to_regclass('{vectorstore_table_name}');
        """)
        table_exists = cursor.fetchone()[0] is not None
        
        cursor.close()
        connection.close()

        return table_exists
    except Exception as e:
        print(f"Error checking if embeddings exist: {e}")
        return False


# Initialize vectorstore only if embeddings don't exist
def store_combined_embeddings(documents, collection_name="combined_data_collection"):
    try:
        vectorstore_table_name = "combined_embeddings"
        
        # Check if embeddings already exist
        if embeddings_exist(vectorstore_table_name):
            print("Embeddings already exist. Skipping embedding storage.")
            # If embeddings exist, load the existing vectorstore
            vectorstore = PGVector.from_existing_vectorstore(
                collection_name=collection_name,
                connection=DB_CONNECTION_URL_2
            )
            return vectorstore
        
        # If embeddings do not exist, create and store new embeddings
        vectorstore = PGVector.from_documents(
            documents,
            embeddings,
            connection=DB_CONNECTION_URL_2,
            collection_name=collection_name
            # No need for postgres_table_name argument
        )
        print("Storing embeddings...")
        return vectorstore
    except Exception as e:
        print(f"Error during embedding storage: {e}")



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

# Retrieve the most relevant document
def get_embedding_response(query, vectorstore):
    try:
       
        results = vectorstore.similarity_search(query, k=1)
        # if results:
        #     return results[0].page_content
        return results
    except Exception as e:
        print(f"Error during similarity search: {e}")
        return None

# Start chat
def start_chat():
    session_id = str(uuid.uuid4())
    chat_history = initialize_chat_history(session_id)

    print("Start chatting (type 'exit' to stop)...")

    insurance_loader = CSVLoader(file_path="insurance.csv")
    happiness_loader = CSVLoader(file_path="world_happiness.csv")

    insurance_documents = insurance_loader.load()
    happiness_documents = happiness_loader.load()

    combined_documents = insurance_documents + happiness_documents

    vector_store = store_combined_embeddings(combined_documents)

    while True:
        query = input("Enter your query: ")
        if query.lower() == "exit":
            print("Exiting chat...")
            break

        # Try embedding-based response
        similar_documents = get_embedding_response(query, vector_store)
      
        prompt = """ You are a helpful assistant and provide the answer for the data that is present in the dataset."""
        try:
            prompt = PromptTemplate.from_template(
                """{context} {query}"""
            )

            chain = prompt | chatgroq_model

            response = chain.invoke({
                "context": similar_documents,
                "query": query
            })

            print("Groq-based Response:", response)
            
            # Insert query and response into chat history
            insert_chat_message(chat_history, query, response)

        except Exception as e:
            print(f"Error during Groq API call: {e}")
            print("AI Response: Sorry, I couldn't process your query.")




if __name__ == "__main__":
    start_chat()
