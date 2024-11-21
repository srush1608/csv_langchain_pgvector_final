import os
import psycopg
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_postgres.vectorstores import PGVector
from dotenv import load_dotenv

load_dotenv()

DB_CONNECTION_URL_2 = os.getenv("DB_CONNECTION_URL_2")

embeddings = GoogleGenerativeAIEmbeddings(
    api_key=os.getenv("GOOGLE_API_KEY"),
    model="models/embedding-001"
)

def store_combined_embeddings(documents, collection_name="combined_data_collection"):
    try:
        vectorstore_table_name = "combined_embeddings"
        
        # Check if embeddings already exist
        if embeddings_exist(vectorstore_table_name):
            print("Embeddings already exist. Skipping embedding storage.")
            vectorstore = PGVector.from_existing_vectorstore(
                collection_name=collection_name,
                connection=DB_CONNECTION_URL_2
            )
            return vectorstore
        
        vectorstore = PGVector.from_documents(
            documents,
            embeddings,
            connection=DB_CONNECTION_URL_2,
            collection_name=collection_name
        )
        print("Storing embeddings...")
        return vectorstore
    except Exception as e:
        print(f"Error during embedding storage: {e}")

def embeddings_exist(vectorstore_table_name="combined_embeddings"):
    try:
        connection = psycopg.connect(DB_CONNECTION_URL_2)
        cursor = connection.cursor()
        
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

def get_embedding_response(query, vectorstore):
    try:
        results = vectorstore.similarity_search(query, k=1)
        return results
    except Exception as e:
        print(f"Error during similarity search: {e}")
        return None
