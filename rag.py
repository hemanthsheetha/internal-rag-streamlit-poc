import os
from openai import OpenAI
from query_pinecone import query_pinecone_multistage_retrieval
import cohere
from llama_index.embeddings.openai import OpenAIEmbedding
import streamlit as st

config =  st.secrets

cohere_api_key = config["COHERE_API_KEY"]
cohere_client = cohere.Client(cohere_api_key)

# Initialize OpenAI API
openai_api_key = config["OPENAI_API_KEY"] 
os.environ["OPENAI_API_KEY"] = openai_api_key

client = OpenAI(api_key=openai_api_key)
if openai_api_key is None:
    raise Exception("No OpenAI API key found. Please set it as an environment variable or in main.py")
 
# Function to generate queries using OpenAI's ChatGPT
def generate_queries_chatgpt(original_query):

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",

        messages=[
            {"role": "system", "content": "You are a helpful question rephrasing GPT that rephrases a given question comprehensively when required"},
            {"role": "user", "content": f"If the question is straightforward then keep it as is. If you think this question needs to be rephrased then rephrase the question to get relevant information and you decide whether to create an in-depth comprehensive report when needed: {original_query}"},
        ],
        temperature=0.4,
    )

    generated_queries = response.choices[0].message.content
    return generated_queries

# Mock function to simulate vector search, returning random scores
def vector_search(query):
    scores_dict = {}
    # distances, indices
    vectors, ids, metadata, scores, texts = query_pinecone_multistage_retrieval(query, top_k=150, top_k_faiss=20)
    for i,score in enumerate(scores):
        scores_dict[score] = texts[i]
    return scores_dict

# Reciprocal Rank Fusion algorithm
def reciprocal_rank_fusion(search_results_dict, k=250): # Tune k
    fused_scores = {}
    for query, doc_scores in search_results_dict.items():
        for rank, (doc, score) in enumerate(sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)):
            if doc not in fused_scores:
                fused_scores[doc] = 0
            previous_score = fused_scores[doc]
            fused_scores[doc] += 1 / (rank + k)
    
    reranked_results = {doc: score for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)}
    return reranked_results

# Dummy function to simulate generative output
def generate_output(reranked_results, queries):
    chunks = list(reranked_results.keys())
    for chunk in chunks:
        if "August" in chunk:
            print("-------------------------------")
            print(chunk)
            print("-------------------------------")
    return f"Final output based on {queries} and reranked documents: {list(reranked_results.keys())[:3]}"

# Function to generate queries using OpenAI's ChatGPT
def generate_answer(original_query, retrieved_context):

    response = client.chat.completions.create(
        model="gpt-4o", 
        messages=[
            {"role": "system", "content": "You are a smart expert in answering queries"},
            {"role": "user", "content": f"Given this query : {original_query} and the following relevant context: {retrieved_context} \n"},
            {"role": "user", "content": "Answer the query in the best possible way and format. The format should be simple. You can also create tables and analyze them when needed. Only answer what is asked. Do not repeat the question/query. Add references(file names) of the context in the end kind of like a research article"}
        ],
        temperature=0.4,
    )

    generated_answer = response.choices[0].message.content
    return generated_answer

def get_rag_response(original_query):
    # original_query = "Who is the policy owner of physical security?"#"Summarize Roles and Responsibilities of Information Security Policy" # "Whats the password policy?" #"What are the steps for HIPAA Breach Procedures for Protected Health Info? Create a comprehensive report with in detail explanation for each step"
    generated_queries = generate_queries_chatgpt(original_query)
    query = generated_queries
    embed_model = OpenAIEmbedding(model="text-embedding-3-small")
    all_results = {}
    query_vector = embed_model.get_text_embedding(query)
    search_results = vector_search(query_vector)
    all_results[query] = search_results
    keys = list(all_results[query].keys())
    docs = list(all_results[query].values())
    rerank_hits = cohere_client.rerank(query=query, documents=list(all_results[query].values()), top_n=100, model="rerank-multilingual-v2.0").results
    retrieved_context = ""
    for chunk in rerank_hits:
        retrieved_context += "--------------"
        retrieved_context+=docs[chunk.index]
        retrieved_context+="--------------"
    
    answer = generate_answer(query, retrieved_context)
    return(answer)
   