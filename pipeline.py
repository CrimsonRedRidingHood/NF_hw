import uuid
from typing import Any
from api_models import StringRequest
from state import model_data
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import ToolMessage

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

    Исключения:
        Нет.
    """
    try:
        config: RunnableConfig = {"configurable": {"thread_id": model_data.session_storage[session_id]}}
        return model_data.graph.invoke({
            "messages": [
                {
                    "role": "user",
                    "content": msg,
                }
            ]
        }, config)['messages'][1:]
    except Exception as e:
        print(f'Error in dispatch_message: {e}')

def pack_answer_from_response(resp : Any, session_id : uuid.UUID):
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
    if not any(type(x) is ToolMessage for x in resp):
        model_data.top_doc_metadata = []
    return {
        "answer" : resp[-1].content,
        "source_documents" : model_data.top_doc_metadata,
        "session_id" : str(session_id)
    }

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
    
    Исключения:
        Нет.
    """
    try:
        msg = req.question
        session_id = uuid.UUID(req.session_id)
        return pack_answer_from_response(dispatch_message(msg, session_id), session_id)
    except Exception as e:
        print(f"Error in process_request_fully: {e}")
