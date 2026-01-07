from flask import Flask, render_template, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
import os
import requests
from collections import deque
from pinecone import Pinecone

app = Flask(__name__)

# Hardcoded Pinecone API Key
PINECONE_API_KEY = "pcsk_3xBdCP_4pWeFB6PgW2tyUg9Amoxc4RwCYLa9h3cPiiyaXcsJzeCTceUuhg5Z6aJGjQkZ7M"
INDEX_NAME = "medicalbot"

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

# ✅ Set API key explicitly in environment
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

# Load embeddings
embeddings = download_hugging_face_embeddings()

# Load Pinecone index
docsearch = PineconeVectorStore.from_existing_index(
    index_name=INDEX_NAME,
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Hardcoded Groq API Key
GROQ_API_KEY = "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"
GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"

# Store last 5 exchanges for chat memory
chat_history = deque(maxlen=5)

def generate_response(query, retrieved_docs):
    """Generate a response using the Groq API with strict topic boundaries."""
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    # Add user input to chat history
    chat_history.append({"role": "user", "content": query})

    # Create conversation history
    messages = [
        {
            "role": "system",
            "content": (
                "You are a specialized medical assistant. Your knowledge is strictly limited to the medical field and the provided context.\n\n"
                
                "*** CRITICAL RULE: OUT OF DOMAIN QUESTIONS ***\n"
                "If the user asks a question that is NOT related to medicine, health, symptoms, or the provided medical context (e.g., questions about celebrities, politicians, coding, general knowledge, history, or personal identities), YOU MUST REFUSE TO ANSWER.\n"
                "Do not use your general knowledge for non-medical topics. \n"
                "Simply reply: 'I don't know, as this does not fall under the medical field.'\n\n"

                "*** INSTRUCTIONS FOR MEDICAL QUESTIONS ***\n"
                "- If the question is medically relevant, use the provided context to give smart, helpful, and accurate responses.\n"
                "- Always be clear, concise, and conversational—like ChatGPT.\n"
                "- Before answering complex cases, ask 2–3 brief follow-up questions to gather details (e.g., symptoms, duration, age) if they are missing.\n"
                "- Once the user responds, provide a direct and informative answer including home remedies or OTC suggestions if applicable.\n"
                "- Do not mention you are an AI. Focus on helping the user.\n"
            )
        }
    ]

    # Append chat history to maintain memory
    messages.extend(chat_history)

    # Add latest query with retrieved context
    messages.append({"role": "user", "content": f"Query: {query}\n\nContext: {context}"})

    payload = {"model": "llama-3.3-70b-versatile", "messages": messages}

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    # Make the request
    response = requests.post(GROQ_ENDPOINT, headers=headers, json=payload)
    
    # --- DEBUGGING ---
    print(f"\n[DEBUG] Status Code: {response.status_code}")
    
    if response.status_code == 200:
        try:
            bot_response = response.json().get("choices", [{}])[0].get("message", {}).get("content", "")
            # Add bot response to chat history
            chat_history.append({"role": "assistant", "content": bot_response})
            return bot_response
        except Exception as e:
            return f"Error parsing JSON: {e}"
    else:
        return f"API Error: {response.status_code} - {response.text}"

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/get", methods=["POST"])
def chat():
    msg = request.form.get("msg", "").strip()
    if not msg:
        return "Error: Empty input"

    print(f"User Input: {msg}")

    retrieved_docs = retriever.invoke(msg)
    response = generate_response(msg, retrieved_docs)

    print(f"Response: {response}")
    return response

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)