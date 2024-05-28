from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from typing import Literal, Dict, Any
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from pydantic import BaseModel, Field
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage

from langchain.schema import Document

import pprint

from langgraph.graph import END, StateGraph



# Set embeddings

os.environ["TAVILY_API_KEY"]="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"
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
print("****************vecroe data base ok *************")

# LLM 

# Data model
class WebSearch(BaseModel):
    """
    The internet. Use web_search for questions that are related to anything else than agents, prompt engineering, and adversarial attacks.
    """
    query: str = Field(description="The query to use when searching the internet.")

class Vectorstore(BaseModel):
    """
    A vectorstore containing documents related to agents, prompt engineering, and adversarial attacks. Use the vectorstore for questions on these topics.
    """
    query: str = Field(description="The query to use when searching the vectorstore.")

# Preamble
preamble = """You are an expert at routing a user question to a vectorstore or web search.
The vectorstore contains documents related to agents, prompt engineering, and adversarial attacks.
Use the vectorstore for questions on these topics. Otherwise, use web-search."""

# Load Hugging Face models
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
question_answerer = pipeline("text-generation", model=model, tokenizer=tokenizer)

def route_question(question: str) -> Literal['web_search', 'vectorstore']:
    """
    Route the question based on its content.
    """
    keywords = ['agent', 'agents', 'prompt engineering', 'adversarial attacks', 'memory']
    if any(keyword in question.lower() for keyword in keywords):
        return 'vectorstore'
    else:
        return 'web_search'

def handle_question(question: str) -> Dict[str, Any]:
    """
    Handle the question by routing it and then invoking the appropriate model.
    """
    route = route_question(question)
    if route == 'web_search':
        # Handle web search question
        # Replace this with actual web search code
        response = question_answerer(f"Web search for: {question}")
    elif route == 'vectorstore':
        # Handle vectorstore question
        # Replace this with actual vectorstore search code
        response = question_answerer(f"Vectorstore search for: {question}")
    else:
        response = "Invalid route"

    return {
        "route": route,
        "response": response
    }

# Example questions
questions = [
    "Who will the Bears draft first in the NFL draft?",
    "What are the types of agent memory?",
    "Hi how are you?"
]

for question in questions:
    result = handle_question(question)
    print(f"Question: {question}")
    print(f"Route: {result['route']}")
    print(f"Response: {result['response']}\n")

print("**************LLM Ok******************")





# Data model
class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""
    binary_score: Literal['yes', 'no']

# Prompt
preamble = """You are a grader assessing relevance of a retrieved document to a user question. \n
If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""

# Load GPT-2 model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

def structured_llm_grader(prompt):
    user_question = prompt["question"]
    document = prompt["document"]
    retrieved_doc_prompt = f"Retrieved document: \n\n {document} \n\n User question: {user_question}"
    
    # Tokenize prompt and generate response
    inputs = tokenizer(retrieved_doc_prompt, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(**inputs, max_length=150)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Convert response to binary score
    binary_score = "yes" if "yes" in response.lower() else "no"
    
    return GradeDocuments(binary_score=binary_score)

# Example question and document
question = "types of agent memory"
document = "Agent memory can be classified into several types, including episodic memory, semantic memory, and procedural memory."

# Grade the document
response = structured_llm_grader({"question": question, "document": document})
print(response)

print("*******************Retrieval Grader************************")



# Preamble
preamble = """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise."""

# Load GPT-2 model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Define function for generating response
def generate_response(prompt):
    inputs = tokenizer.encode(prompt, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(inputs, max_length=150, num_return_sequences=1)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Prompt
def prompt(x):
    return f"Question: {x['question']} \nAnswer: "

# Chain (define the sequence of functions)
rag_chain = [prompt, generate_response, StrOutputParser()]

# Run
x = {"documents": docs, "question": question}
for func in rag_chain:
    if isinstance(func, StrOutputParser):
        x = func.parse(x)
    else:
        x = func(x)

print(x)


print("*********************Generator Ok***************************")





# Define Hugging Face GPT-2 pipeline
chat_pipeline = pipeline("text-generation", model="gpt2")

# Define prompt function
def prompt(x):
    return f"User question: \n\n {x['question']} \n\n LLM generation: {x['generation']}"

# Define llm_chain as a function
def llm_chain(x):
    x['generation'] = chat_pipeline(x['question'])
    return x

# Instantiate StrOutputParser
output_parser = StrOutputParser()

# Run
question = "Hi how are you?"
generation = llm_chain({"question": question})

# Print the output
print(prompt(generation))



print("********************LLM fallback Ok **************************")


# Data model
class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""
    binary_score: str = Field(description="Answer is grounded in the facts, 'yes' or 'no'")

# Preamble
preamble = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n
Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""

# Define Hugging Face GPT-2 pipeline
chat_pipeline = pipeline("text-generation", model="gpt2")

# Prompt
def prompt(x):
    return f"Set of facts: \n\n {x['documents']} \n\n LLM generation: {x['generation']}"

# Chain (define the sequence of functions)
hallucination_chain = [prompt, chat_pipeline, StrOutputParser()]

# Run
docs = "Your retrieved facts here"
generation = "The LLM generation here"
x = {"documents": docs, "generation": generation}
for func in hallucination_chain:
    if isinstance(func, StrOutputParser):
        x = func.parse(x)
    else:
        x = func(x)

print(x)


print("********************************Hallucination Grader********************************")


# Data model
class GradeAnswer(BaseModel):
    """Binary score to assess answer addresses question."""
    binary_score: str = Field(description="Answer addresses the question, 'yes' or 'no'")

# Preamble
preamble = """You are a grader assessing whether an answer addresses / resolves a question \n
Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question."""

# Define Hugging Face GPT-2 pipeline
chat_pipeline = pipeline("text-generation", model="gpt2")

# Prompt
def prompt(x):
    return f"User question: \n\n {x['question']} \n\n LLM generation: {x['generation']}"

# Chain (define the sequence of functions)
answer_chain = [prompt, chat_pipeline, StrOutputParser()]

# Run
question = "User's question here"
generation = "LLM generation here"
x = {"question": question, "generation": generation}
for func in answer_chain:
    if isinstance(func, StrOutputParser):
        x = func.parse(x)
    else:
        x = func(x)

print(x)
print("*************************Answer Grader*****************************")

from langchain_community.tools.tavily_search import TavilySearchResults

web_search_tool = TavilySearchResults()

print("******************web search Ok ********************")



from typing_extensions import TypedDict
from typing import List


class GraphState(TypedDict):
    """|
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        documents: list of documents
    """

    question: str
    generation: str
    documents: List[str]


    print("************************Graph state Ok**************************")


def retrieve(state):
    print("---RETRIEVE---")
    question = state["question"]

    # Retrieval
    # Placeholder for retriever implementation
    documents = retriever(question)
    return {"documents": documents, "question": question}

def llm_fallback(state):
    print("---LLM Fallback---")
    question = state["question"]
    # Placeholder for llm_chain implementation
    generation = llm_chain({"question": question})
    return {"question": question, "generation": generation}

def generate(state):
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    if not isinstance(documents, list):
        documents = [documents]

    # Placeholder for rag_chain implementation
    generation = rag_chain({"documents": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}

def grade_documents(state):
    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]

    # Score each doc
    filtered_docs = []
    for d in documents:
        # Placeholder for retrieval_grader implementation
        score = response({"question": question, "document": d.page_content})
        grade = score.binary_score
        if grade == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            continue
    return {"documents": filtered_docs, "question": question}

def web_search(state):
    print("---WEB SEARCH---")
    question = state["question"]

    # Placeholder for web_search_tool implementation
    docs = web_search_tool.invoke({"query": question})
    web_results = "\n".join([d["content"] for d in docs])
    web_results = Document(page_content=web_results)

    return {"documents": web_results, "question": question}

### Edges ###

def route_question(state):
    print("---ROUTE QUESTION---")
    question = state["question"]
    sources = question_answerer( question)

    if not sources:
        print("---ROUTE QUESTION TO LLM---")
        return "llm_fallback"

    # Iterate over potential sources
    for source in sources:
        if "tool_calls" not in source:
            print("---ROUTE QUESTION TO LLM---")
            return "llm_fallback"

        # Choose the first available tool call
        tool_calls = source["tool_calls"]
        if tool_calls:
            datasource = tool_calls[0]["function"]["name"]
            if datasource == "web_search":
                print("---ROUTE QUESTION TO WEB SEARCH---")
                return "web_search"
            elif datasource == "vectorstore":
                print("---ROUTE QUESTION TO RAG---")
                return "generate"

    print("---ROUTE QUESTION TO LLM---")
    return "llm_fallback"


def decide_to_generate(state):
    print("---ASSESS GRADED DOCUMENTS---")
    question = state["question"]
    filtered_documents = state["documents"]

    if not filtered_documents:
        print("---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, WEB SEARCH---")
        return "web_search"
    else:
        print("---DECISION: GENERATE---")
        return "generate"

def grade_generation_v_documents_and_question(state):
    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    # Placeholder for hallucination_grader implementation
    score = hallucination_chain({"documents": documents, "generation": generation})
    grade = score.binary_score

    if grade == "yes":
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        score = answer_chain({"question": question, "generation": generation})
        grade = score.binary_score
        if grade == "yes":
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"
    else:
        print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        return "not supported"






print("*****************************ALl Ok**************************************")




workflow = StateGraph(GraphState)

# Define the nodes
workflow.add_node("web_search", web_search)  # web search
workflow.add_node("retrieve", retrieve)  # retrieve
workflow.add_node("grade_documents", grade_documents)  # grade documents
workflow.add_node("generate", generate)  # rag
workflow.add_node("llm_fallback", llm_fallback)  # llm

# Build graph
workflow.set_conditional_entry_point(
    route_question,
    {
        "web_search": "web_search",
        "vectorstore": "retrieve",
        "llm_fallback": "llm_fallback",
    },
)
workflow.add_edge("web_search", "generate")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "web_search": "web_search",
        "generate": "generate",
    },
)
workflow.add_conditional_edges(
    "generate",
    grade_generation_v_documents_and_question,
    {
        "not supported": "generate",  # Hallucinations: re-generate
        "not useful": "web_search",  # Fails to answer question: fall-back to web-search
        "useful": END,
    },
)
workflow.add_edge("llm_fallback", END)

# Compile
app = workflow.compile()


print("***************dfgdfgdfgdf*****************")




inputs = {
    "question": "What player are the Bears expected to draft first in the 2024 NFL draft?"
}
for output in app.stream(inputs):
    for key, value in output.items():
        # Node
        pprint.pprint(f"Node '{key}':")
        # Optional: print full state at each node
    pprint.pprint("\n---\n")

# Final generation
pprint.pprint(value["generation"])