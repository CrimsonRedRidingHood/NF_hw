from dotenv import load_dotenv
from langchain_classic.tools.retriever import create_retriever_tool
from langchain_ollama import OllamaEmbeddings
from typing import List
from langchain_core.runnables import RunnableConfig
import uuid
from typing import Optional, Dict
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage
from langchain_community.tools import DuckDuckGoSearchResults
from fastapi import FastAPI
from pydantic import BaseModel
from llms import *
from vector_store import *
from model import *

load_dotenv()

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
    """
    Прогоняет сообщение через LLM-граф, используя состояние графа,
    соответствующее полученному от пользователя session_id

    Входные данные:
        msg - сообщение для обработки
        session_id - ID сессии с пользователем
    
    Выходные данные:
        Цепочка обработки сообщения пользователя моделью,
        по шагам - тексты каждого запроса в процессе рассуждения модели,
        а также информация по вызовам тулов - их аргументы и результаты.
    """
    config: RunnableConfig = {"configurable": {"thread_id": session_storage[session_id]}}
    return graph.invoke({
        "messages": [
            {
                "role": "user",
                "content": msg,
            }
        ]
    }, config)['messages'][1:]

def pack_answer_from_response(resp, session_id):
    """
    Сохранение результатов обработки запроса в словарь для дальнейшей
    запаковки в response и отправки пользователю

    Входные данные:
        resp - результат обработки запроса
        session_id - ID сеанса с пользователем
    
    Выходные данные:
        Словарь формата:
            answer: str
            source_documents: List[{"source": str, "snippet": str}]
            session_id: str
    """
    global model_data

    if not any(type(x) is ToolMessage for x in resp):
        model_data.top_doc_metadata = []
    return {
        "answer" : resp[-1].content,
        "source_documents" : model_data.top_doc_metadata,
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
    """
    Функция обработки строки-запроса и получения
    словаря с заполненными для формирования строки-ответа полями

    Входные данные:
        req: StringRequest - строка-запрос, содержащая
                сам вопрос, а также ID сеанса с пользователем
    
    Выходные данные:
        Совпадают с pack_answer_from_response, т.е.
        Словарь формата:
            answer: str
            source_documents: List[{"source": str, "snippet": str}]
            session_id: str
    """
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

COLLECTION_NAME = "nf_hw_collection"

vector_db = load_vector_store(COLLECTION_NAME, embeddings)

@tool(description="Возвращает ближайшие по смыслу вхождения текстов из базы")
def retrieval_function(x, k = 9, filter: Optional[Dict[str, str]] = None, **kwargs):
    """
    Функция-ретривер.

    Возвращает наиболее подходящие к `x` входящие в векторную БД
    записи. Количество возвращаемх записей определяется параметром k.

    Входные данные:
        См. параметры similarity_search, т.е.:
            x - текст для поиска в БД
            k - количество top-k возвращаемых записей
    """
    global model_data
    retval = vector_db.similarity_search(x, k, filter, **kwargs)[4:]
    model_data.top_doc_metadata = [{"source": d.metadata['source'], "snippet": d.page_content[:30] + ('' if len(d.page_content) <= 30 else '...')} for d in retval]
    return retval

retriever_tool = create_retriever_tool(
    retrieval_function,
    "retrieve_neoflex_info",
    "Найти и вернуть информацию об организации 'Неофлекс'."
)

USED_MODEL = "GPT-4o"
USE_LOCAL_MODEL = False
llm = create_llm(USE_LOCAL_MODEL, USED_MODEL)

model_data.set_parameters(llm, retriever_tool)

graph = create_graph()

session_storage = SessionStorage()

app = FastAPI(title="String Processor")

@app.post("/process-string", response_model=StringResponse)
async def process_string(request: StringRequest):
    """
    Функция обработки endpoint'а "process-string".
    Принимает на вход строку, содержащую "question" и "session_id",
    Возвращает результат обработки "question" через LLM-граф.

    Входные данные:
        request - строка запроса

    Выходные данные:
        StringResponse - строка-ответ на запрос, содержащая
        "answer", "source_documents" и "session_id"
    """
    
    print(f"Received request: {request}")
    result = process_request_fully(request)
    print(f"Result: {result}")
    
    return StringResponse(answer=result['answer'], source_documents=result['source_documents'], session_id=result['session_id'])

@app.get("/")
async def root():
    """
    Функция обработки endpoint'а для проверки, что сервер запущен и обработка запросов работает.
    """
    return {"message": "Server is up!"}

# Run with: uvicorn [filename]:app --reload