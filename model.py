from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
import os
import getpass
from dotenv import load_dotenv
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_classic.tools.retriever import create_retriever_tool
import json
from langchain_ollama import OllamaEmbeddings
from langgraph.graph import MessagesState
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from typing import List
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.runnables import RunnableConfig
import uuid
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_community.tools import DuckDuckGoSearchResults
from fastapi import FastAPI
from pydantic import BaseModel
from tqdm import tqdm
from langchain_chroma import Chroma


DOCS_FILENAME = 'prepared_documents_concat.json'

def split_text_by_chunks(json_objs):
    result = []
    
    for obj in json_objs:
        url = obj['url']
        section = obj['section']
        text = obj['text']
        
        current_text = text
        chunk_count = 0
        
        while current_text:
            chunk_count += 1
            
            if len(current_text) <= 500:
                new_obj = {
                    'url': url,
                    'section': f"{section}_part{chunk_count}",
                    'text': current_text.strip()
                }
                result.append(new_obj)
                break
            
            period_pos = current_text.find('.', 500)
            
            if period_pos == -1:
                new_obj = {
                    'url': url,
                    'section': f"{section}_part{chunk_count}",
                    'text': current_text.strip()
                }
                result.append(new_obj)
                break
            else:
                split_pos = period_pos + 1
                chunk = current_text[:split_pos].strip()
                
                new_obj = {
                    'url': url,
                    'section': f"{section}_part{chunk_count}",
                    'text': chunk
                }
                result.append(new_obj)
                
                current_text = current_text[split_pos:].strip()
    
    return result

def generate_query_or_respond(state: MessagesState):
    """Call the model to generate a response based on the current state. Given
    the question, it will decide to retrieve using the retriever tool, or simply respond to the user.
    """
    response = (
        llm
        .bind_tools([retriever_tool]).invoke(state["messages"])  
    )
    return {"messages": [response]}

GENERATE_PROMPT = (
    "Ты - ассистент для консультации по разным вопросам, в частности - касающихся компании Неофлекс. "
    "Используй приведённый ниже контекст, чтобы ответить на вопрос. Возможно, но не точно, ответ уже содержится в контексте. "
    "Если к ответу вообще никак не получается прийти, просто сообщи, что не знаешь ответа. "
    "Вопрос: {question} \n"
    "Контекст: {context}"
)

def generate_answer(state: MessagesState):
    """Generate an answer."""
    #print(f"generate answer called: {state}")
    for msg in reversed(state["messages"]):
        if type(msg) is HumanMessage:
            break
    question = msg.content
    context = state["messages"][-1].content
    prompt = GENERATE_PROMPT.format(question=question, context=context)
    print(f"generate_answer called with prompt={prompt}")
    response = llm.invoke([{"role": "user", "content": prompt}])
    return {"messages": [response]}

def dispatch_message_streamed(msg: str):
    result = ""
    for chunk in graph.stream(
    {
        "messages": [
            {
                "role": "user",
                "content": msg,
            }
        ]
    }):
        for node, update in chunk.items():
            print("Update from node", node)
            result = update["messages"][-1]
            update["messages"][-1].pretty_print()
            print('\n\n')
    
    return result

class SessionStorage:
    def __init__(self):
        self.d = {}
        self.idx = 1
    
    def __getitem__(self, key):
        if key not in self.d:
            self.d[key] = self.idx
            self.idx += 1
            return self.idx - 1
        else:
            return self.d[key]

def dispatch_message(msg: str, session_id: uuid):
    config: RunnableConfig = {"configurable": {"thread_id": session_storage[session_id]}}
    return graph.invoke({
        "messages": [
            {
                "role": "user",
                "content": msg,
            }
        ]
    }, config)['messages'][1:]

top_doc_metadata = []

def pack_answer_from_response(resp, session_id):
    global top_doc_metadata

    if not any(type(x) is ToolMessage for x in resp):
        top_doc_metadata = []
    return {
        "answer" : resp[-1].content,
        "source_documents" : top_doc_metadata,
        "session_id" : str(session_id)
    }

# Define request/response models
class StringRequest(BaseModel):
    session_id: str
    question: str

class SourceDoc(BaseModel):
    source: str
    snippet: str

class StringResponse(BaseModel):
    answer: str
    source_documents: List[SourceDoc]
    session_id: str

def process_request_fully(req: StringRequest):
    try:
        msg = req.question
        session_id = uuid.UUID(req.session_id)
        return pack_answer_from_response(dispatch_message(msg, session_id), session_id)
    except Exception as e:
        print(f"Error in process_request_fully: {e}")

embeddings = OllamaEmbeddings(
    model="qwen3-embedding",
)

search = DuckDuckGoSearchResults()

vector_store = Chroma(
    collection_name="nf_hw_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not necessary
)

if len(vector_store.get()['ids']) == 0:
    with open(DOCS_FILENAME, 'r', encoding='utf-8') as file:
        docs = json.load(file)

    for doc in tqdm(split_text_by_chunks(docs)):
        vector_store.add_documents(documents=[Document(page_content=doc['text'], metadata={"source": doc['url']})])

@tool(description="Возвращает ближайшие по смыслу вхождения текстов из базы")
def retrieval_function(x, k = 9, **kwargs):
    global top_doc_metadata
    retval = vector_store.similarity_search(x, k, **kwargs)[4:]
    top_doc_metadata = [{"source": d.metadata['source'], "snippet": d.page_content[:30] + ('' if len(d.page_content) <= 30 else '...')} for d in retval]
    return retval

retriever_tool = create_retriever_tool(
    retrieval_function,
    "retrieve_neoflex_info",
    "Найти и вернуть информацию об организации 'Неофлекс'."
)

load_dotenv()

template = """Question: {question}
Answer: Let's think step by step."""

model_data = {
    "GPT-4o" :
    {
        "tokens" : 40,
        "name" : "openai/gpt-4o"
    },
    "DeepSeek" :
    {
        "tokens" : 100,
        "name" : "tngtech/deepseek-r1t2-chimera:free"
    },
    "Grok":
    {
        "tokens": 100,
        "name" : "x-ai/grok-4.1-fast"
    },
    "Nemotron":
    {
        "tokens": 200,
        "name" : "nvidia/nemotron-nano-12b-v2-vl:free"
    }
}

USED_MODEL = "GPT-4o"
USE_LOCAL_MODEL = False

if not USE_LOCAL_MODEL:
    if not os.environ.get("OPENROUTER_API_KEY"):
        os.environ["OPENROUTER_API_KEY"] = getpass.getpass("Enter API key for OpenRouter: ")

    llm = ChatOpenAI(
    max_tokens=model_data[USED_MODEL]["tokens"],
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
    model=model_data[USED_MODEL]["name"],
    default_headers={
        "HTTP-Referer": "YOUR_SITE_URL", # Optional. Site URL for rankings on openrouter.ai.
        "X-Title": "YOUR_SITE_NAME", # Optional. Site title for rankings on openrouter.ai.
    }
    )
else:
    llm = ChatOllama(
        model="gpt-oss:20b",
        temperature=0,
    )

workflow = StateGraph(MessagesState)

# Define the nodes we will cycle between
workflow.add_node(generate_query_or_respond)
workflow.add_node("retrieve", ToolNode([retriever_tool]))
workflow.add_node(generate_answer)

workflow.add_edge(START, "generate_query_or_respond")
workflow.add_conditional_edges(
    "generate_query_or_respond",
    # Assess LLM decision (call `retriever_tool` tool or respond to the user)
    tools_condition,
    {
        # Translate the condition outputs to nodes in our graph
        "tools": "retrieve",
        END: END,
    },
)

workflow.add_edge("retrieve", "generate_answer")
workflow.add_edge("generate_answer", END)

graph = workflow.compile()

checkpointer = InMemorySaver()
graph = workflow.compile(checkpointer=checkpointer)

session_storage = SessionStorage()

app = FastAPI(title="String Processor")

@app.post("/process-string", response_model=StringResponse)
async def process_string(request: StringRequest):
    """
    Accepts a string and returns a processed version.
    This example reverses the string and adds a prefix.
    """
    
    print(f"Received request: {request}")
    result = process_request_fully(request)
    print(f"Result: {result}")
    
    return StringResponse(answer=result['answer'], source_documents=result['source_documents'], session_id=result['session_id'])

@app.get("/")
async def root():
    return {"message": "Server is up!"}

# Run with: uvicorn [filename]:app --reload