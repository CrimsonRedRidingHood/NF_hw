from fastapi import FastAPI
from config import USE_LOCAL_MODEL, USED_MODEL, COLLECTION_NAME, embeddings
from state import model_data
from api_models import StringRequest, StringResponse
from llms import create_llm
from model import create_graph
from tools import retriever_tool
from vector_store import load_vector_store
from session_storage import SessionStorage
from pipeline import process_request_fully

vector_db = load_vector_store(COLLECTION_NAME, embeddings)

llm = create_llm(USE_LOCAL_MODEL, USED_MODEL)

graph = create_graph(retriever_tool)

session_storage = SessionStorage()

model_data.set_parameters(llm, retriever_tool, vector_db, session_storage, graph)

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