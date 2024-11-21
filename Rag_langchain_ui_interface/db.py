import os
import psycopg
from dotenv import load_dotenv
from langchain_community.document_loaders.csv_loader import CSVLoader

load_dotenv()

DB_CONNECTION_URL_2 = os.getenv("DB_CONNECTION_URL_2")

def load_documents():
    insurance_loader = CSVLoader(file_path="insurance.csv")
    happiness_loader = CSVLoader(file_path="world_happiness.csv")

    insurance_documents = insurance_loader.load()
    happiness_documents = happiness_loader.load()

    print(f"Loaded {len(insurance_documents)} insurance documents and {len(happiness_documents)} happiness documents.")
    return insurance_documents, happiness_documents
