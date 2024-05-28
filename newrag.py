from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
import os
from langchain_community.embeddings import HuggingFaceEmbeddings
# Set embeddings

# Docs to index
urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]
# Load
docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]
# Split
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=512, chunk_overlap=0
)
doc_splits = text_splitter.split_documents(docs_list)


model_kwargs={"device":"cuda"}
encode_kwargs={"normalize_embeddings":True}
model_name = "mixedbread-ai/mxbai-embed-large-v1"

embd = HuggingFaceEmbeddings(model_name=model_name,model_kwargs=model_kwargs,encode_kwargs=encode_kwargs)
# Add to vectorstore
vectorstore = Chroma.from_documents(
    documents=doc_splits,
    embedding=embd,
)

retriever = vectorstore.as_retriever()
print(retriever)


template = """Answer the question based only on the following context:
{context}

Question: {question}
"""



context = "Your provided context goes here."
question = "Your question goes here."

# Fill in the template
formatted_prompt = template.format(context=context, question=question)

# Now you can use this `formatted_prompt` with a Hugging Face model.

from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the model and tokenizer
model_name = "gpt-3.5-turbo"  # Replace with your desired model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Encode the input prompt
inputs = tokenizer.encode(formatted_prompt, return_tensors="pt")

# Generate a response
outputs = model.generate(inputs, max_length=500)

# Decode and print the response
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)

