import os
import time

from src.helper import load_pdf_file, text_split, download_hugging_face_embeddings
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore  # Updated import

load_dotenv()

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "carebot"

# Create index if needed
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    print(f"[INFO] Created index '{index_name}'")
    time.sleep(60)
else:
    print(f"[INFO] Index '{index_name}' already exists")

# Load data
extracted_data = load_pdf_file(data='Data/')
text_chunks = text_split(extracted_data)
print(f"[INFO] Extracted {len(text_chunks)} text chunks")

# Get embeddings
embeddings = download_hugging_face_embeddings()

# Upload in batches of 200
batch_size = 200
for i in range(0, len(text_chunks), batch_size):
    batch = text_chunks[i:i + batch_size]
    PineconeVectorStore.from_documents(
        documents=batch,
        embedding=embeddings,
        index_name=index_name
    )
    print(f"[INFO] Uploaded batch {i // batch_size + 1}")

# Verify count
index = pc.Index(index_name)
stats = index.describe_index_stats()
print(f"[INFO] Total vectors stored: {stats['total_vector_count']}")