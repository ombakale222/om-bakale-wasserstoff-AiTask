from flask import Flask, request, jsonify
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
from torch.nn.functional import cosine_similarity

app = Flask(__name__)

# Initialize embeddings
embedding_model_name = "mixedbread-ai/mxbai-embed-large-v1"
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": True}
embd = HuggingFaceEmbeddings(model_name=embedding_model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs)

# URLs of the documents to index
urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]
docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

# Split documents into smaller chunks
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=512, chunk_overlap=0)
doc_splits = text_splitter.split_documents(docs_list)

# Add document chunks to the vector store with embeddings
vectorstore = Chroma.from_documents(doc_splits, embedding=embd)
V_retriever = vectorstore.as_retriever()

# Initialize GPT-2 tokenizer and model
gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')

def retrieve(query, k=5):
    # Convert user query to vector representation
    query_vector = embd.embed_query(query)
    
    # Retrieve chunks and calculate similarity scores
    retrieved_chunks = V_retriever.invoke(query)
    
    # Filter out chunks without vectors
    retrieved_chunks = [chunk for chunk in retrieved_chunks if hasattr(chunk, 'vector')]
    
    # If there are no chunks left, return an empty list
    if not retrieved_chunks:
        return []
    
    chunk_vectors = torch.stack([chunk.vector for chunk in retrieved_chunks])
    similarity_scores = cosine_similarity(query_vector.unsqueeze(0), chunk_vectors)
    
    # Get top K chunks with highest similarity scores
    top_indices = similarity_scores.squeeze(0).argsort(descending=True)[:k]
    retrieved_docs = [retrieved_chunks[i] for i in top_indices]
    
    return retrieved_docs

def generate_response(query, retrieved_docs, max_length=512, max_new_tokens=50):
    # Combine user query and retrieved chunks using a prompt template
    prompt = f"{query}\n\n"
    prompt += "\n".join([doc.page_content for doc in retrieved_docs])
    
    # Generate response based on augmented prompt
    inputs = gpt2_tokenizer.encode(prompt, return_tensors='pt', max_length=max_length, truncation=True)
    outputs = gpt2_model.generate(inputs, max_length=max_length, max_new_tokens=max_new_tokens, num_return_sequences=1)
    response = gpt2_tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return response

def rag_system(query, k=5):
    retrieved_docs = retrieve(query, k)
    response = generate_response(query, retrieved_docs)
    return response

def develop_reasoning_steps(initial_response, previous_context):
    # Placeholder function for developing reasoning steps
    return [initial_response]

def refine_response_based_on_thought_steps(thought_steps):
    # Placeholder function for refining response based on thought steps
    return thought_steps[-1]

def process_query_with_chain_of_thought(user_query, previous_context):
    initial_response = rag_system(user_query)
    thought_steps = develop_reasoning_steps(initial_response, previous_context)
    
    # Check if there are thought steps
    if thought_steps:
        final_response = refine_response_based_on_thought_steps(thought_steps)
    else:
        final_response = "Sorry, I can't provide an answer to that question at the moment."
    
    return final_response

@app.route('/', methods=['GET'])
def index():
    return 'Welcome to the RAG-based QA system!'

@app.route('/process_query', methods=['POST'])
def process_query():
    data = request.json
    user_query = data['user_query']
    previous_context = data.get('previous_context', "")
    
    final_response = process_query_with_chain_of_thought(user_query, previous_context)
    return jsonify({'final_response': final_response})

if __name__ == '__main__':
    app.run(debug=True)
