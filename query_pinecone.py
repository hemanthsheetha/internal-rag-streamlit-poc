from pinecone import Pinecone
import streamlit as st

config =  st.secrets

# Initialize Pinecone
pc = Pinecone(api_key=config["PINECONE_API_KEY"])#, environment='YOUR_ENVIRONMENT')  # replace with your actual API key and environment
index_name = 'lner-no-zarchive-no-excel-with-isms' #'soc2-policies-markdown-index' #'direxion-compliance-worker'

# Connect to the Pinecone index
index = pc.Index(index_name)

# Define a function to query Pinecone for vectors
def query_pinecone_multistage_retrieval(query_vector, top_k=10, top_k_faiss=5):
    # Perform a query on the Pinecone index
    response = index.query(
        vector=query_vector,
        top_k=top_k,
        include_values=True,  # Retrieve vectors along with metadata
        include_metadata=True
    )
    
    k = top_k_faiss
    # Extract the vectors and their corresponding metadata
    retrieved_vectors = [match['values'] for match in response['matches']]
    ids = [match['id'] for match in response['matches']]
    metadata = [match['metadata'] for match in response['matches']]
    scores = [match['score'] for match in response['matches']]
    texts = [match['text'] for match in metadata]

    
    return retrieved_vectors, ids, metadata, scores, texts 