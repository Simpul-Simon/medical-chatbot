from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
import os
import jwt
import bcrypt
from datetime import datetime, timedelta
from functools import wraps
import pymysql
from flask_cors import CORS  # Added CORS import
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.prompts import PromptTemplate
from pinecone import Pinecone, ServerlessSpec

from src.prompt import prompt_template
from src.helper import download_hugging_face_embeddings

from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate

# Load environment variables
load_dotenv()

# Flask app
app = Flask(__name__)

# Enable CORS for all routes to handle cross-origin requests from the React frontend
CORS(app)

# Load keys
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
HUGGINGFACEHUB_API_TOKEN = os.getenv('HUGGINGFACEHUB_API_TOKEN')
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN

# MySQL connection setup
db = pymysql.connect(
    host=os.getenv('MYSQL_HOST'),
    user=os.getenv('MYSQL_USER'),
    password=os.getenv('MYSQL_PASSWORD'),
    database=os.getenv('MYSQL_DB')
)
cursor = db.cursor()

# JWT secret key and expiration
JWT_SECRET_KEY = os.getenv('JWT_SECRET_KEY')
JWT_EXPIRATION_DELTA = timedelta(days=1)

# Pinecone initialization
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "carebot"
index = pc.Index(index_name)

# Embedding model
embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Vector store setup using the new Pinecone client
vectorstore = PineconeVectorStore(
    index=index,
    embedding=embed_model,
    text_key="text"
)

# Retriever
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# LLM setup
llm = ChatGroq(
    temperature=0.5,
    model_name="llama3-70b-8192",
    api_key=os.getenv("GROQ_API_KEY")
)

# Prompt template
prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(prompt_template),
    HumanMessagePromptTemplate.from_template("{input}")
])

# Chains
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# Optional: RetrievalQA (used in "/get" route)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt},
    retriever=retriever,
    return_source_documents=True
)

# Helper function to verify JWT
def token_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        token = None
        if 'Authorization' in request.headers:
            token = request.headers['Authorization'].split(" ")[1]
        if not token:
            return jsonify({'message': 'Token is missing!'}), 403
        try:
            data = jwt.decode(token, JWT_SECRET_KEY, algorithms=["HS256"])
            current_user = data['email']
        except:
            return jsonify({'message': 'Token is invalid!'}), 403
        return f(current_user, *args, **kwargs)
    return decorated_function

# User signup route
@app.route('/signup', methods=['POST'])
def signup():
    data = request.get_json()
    username = data.get('username')
    email = data.get('email')
    password = data.get('password')

    # Check if user already exists
    cursor.execute("SELECT * FROM users WHERE email = %s", (email,))
    user = cursor.fetchone()
    if user:
        return jsonify({'message': 'User already exists'}), 400

    # Hash the password
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

    # Insert new user into the database
    cursor.execute("INSERT INTO users (username, email, password) VALUES (%s, %s, %s)",
                   (username, email, hashed_password))
    db.commit()

    return jsonify({'message': 'User created successfully'}), 201

# User login route
@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')

    cursor.execute("SELECT * FROM users WHERE email = %s", (email,))
    user = cursor.fetchone()

    if not user:
        return jsonify({'message': 'User not found'}), 400

    stored_password = user[3]  # Assuming the password is in the 4th column
    if not bcrypt.checkpw(password.encode('utf-8'), stored_password.encode('utf-8')):
        return jsonify({'message': 'Invalid password'}), 400

    # Generate JWT token
    token = jwt.encode({
        'email': email,
        'exp': datetime.utcnow() + JWT_EXPIRATION_DELTA
    }, JWT_SECRET_KEY, algorithm="HS256")

    return jsonify({'token': token}), 200

# Meta query check
def is_bot_meta_query(user_input: str) -> bool:
    meta_keywords = [
        "who are you", "hi", "hello", "your name", "what are you", "what is your purpose",
        "how do you work", "are you human", "what can you do"
    ]
    return any(kw in user_input.lower() for kw in meta_keywords)

def handle_meta_query() -> str:
    return (
        "Hi! 😊 I'm **CareBot**, your friendly AI health assistant. "
        "I help answer your health-related questions using trusted medical sources. "
        "Feel free to ask me about symptoms, conditions, treatments, or healthy habits!"
    )

# Chat route (for CareBot responses)
@app.route("/get", methods=["POST"])
@token_required
def get_bot_response(current_user):
    data = request.get_json()
    user_input = data.get("msg", "")

    if is_bot_meta_query(user_input):
        response = handle_meta_query()
    else:
        try:
            docs = retriever.invoke(user_input)
            result = question_answer_chain.invoke({"context": docs, "input": user_input})
            response = result
        except Exception as e:
            print(f"Error processing message: {e}")
            response = "An error occurred while generating a response."

    return jsonify({"response": str(response)})


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
