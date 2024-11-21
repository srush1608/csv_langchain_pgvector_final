from flask import Flask, render_template, request, jsonify
from embeddings import store_combined_embeddings, get_embedding_response
from chat_history import initialize_chat_history, insert_chat_message
from db import load_documents
import os

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('chat.html')

@app.route('/chat', methods=['POST'])
def chat():
    query = request.form['query']
    print(f"User Query: {query}")

    # Load documents for embedding
    insurance_documents, happiness_documents = load_documents()
    
    combined_documents = insurance_documents + happiness_documents
    vector_store = store_combined_embeddings(combined_documents)

    similar_documents = get_embedding_response(query, vector_store)
    print(f"Similar Documents: {similar_documents}")

    prompt = """ You are a helpful assistant and provide the answer for the data that is present in the dataset."""
    try:
        from langchain_core.prompts import PromptTemplate
        from langchain_groq import ChatGroq

        prompt_template = PromptTemplate.from_template(
            """{context} {query}"""
        )

        chatgroq_model = ChatGroq(
            model="mixtral-8x7b-32768",
            temperature=0.0,
            max_retries=2
        )

        chain = prompt_template | chatgroq_model

        response = chain.invoke({
            "context": similar_documents,
            "query": query
        })

        # Check if the response is an AIMessage object
        if hasattr(response, 'content'):
            ai_response_content = response.content  # Extract content from AIMessage
        else:
            ai_response_content = str(response)  # Fallback to string representation

        print(f"AI Response: {ai_response_content}")

        chat_history = initialize_chat_history()
        insert_chat_message(chat_history, query, ai_response_content)

        return jsonify({'response': ai_response_content})
    except Exception as e:
        print(f"Error during Groq API call: {e}")
        return jsonify({'response': "Sorry, I couldn't process your query."})

if __name__ == "__main__":
    app.run(debug=True)

