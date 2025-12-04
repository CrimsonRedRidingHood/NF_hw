from langgraph.graph import MessagesState
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import HumanMessage
from llms import *
import logging

logging.basicConfig(level=logging.DEBUG)

class ModelData:
    _instance = None
    _initialized = False
    llm = None
    top_doc_metadata = []
    retriever_tool = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self._initialized = True

    def set_parameters(self, llm, retriever_tool):
        """
        Установка параметров графа - используемая LLM
        и тул для получения наиболее близких по смыслу
        текстов из БД
        """
        self.llm = llm
        self.retriever_tool = retriever_tool

model_data = ModelData()
    
def generate_query_or_respond(state: MessagesState):
    """
    Определить, нужно ли вызывать tool и, либо вызвать его
    и вернуть результат вызова, либо обратиться к LLM-модели
    за генерацией ответа, если tool вызывать не нужно.

    Входные данные:
        state: MessageState - обрабатываемый запрос

    Выходные данные:
        Словарь "messages": List[Union[ToolResponse, AIResponse]] -
            результат обработки запроса
    """
    response = (
        model_data.llm
        .bind_tools([model_data.retriever_tool]).invoke(state["messages"])  
    )
    logging.debug(f"generate_query_or_respond returns {response}")
    return {"messages": [response]}

GENERATE_PROMPT = (
    "Ты - ассистент для консультации по разным вопросам, в частности - касающихся компании Неофлекс. "
    "Используй приведённый ниже контекст, чтобы ответить на вопрос. Возможно, но не точно, ответ уже содержится в контексте. "
    "Если к ответу вообще никак не получается прийти, просто сообщи, что не знаешь ответа. "
    "Не используй табличное форматирование и особые Markdown-стилизации, используй обычное текстовое представление, это важно."
    "Вопрос: {question} \n"
    "Контекст: {context}"
)

def generate_answer(state: MessagesState):
    """
    Функция генератии ответа LLM на запрос по шаблону выше (GENERATE_PROMPT)
    с использованием последнего элемента цепочки state как контекста
    и первого с конца элемента цепочки, имеющего тип HumanMessage в качестве
    вопроса. ФУНКЦИЯ НЕ ДОЛЖНА ВЫЗЫВАТЬСЯ КАК ПЕРВЫЙ ЭЛЕМЕНТ ГРАФА, иначе
    вопрос и контекст могут совпадать.

    Входные данные:
        state: MessagesState - цепочка обработки запроса

    Выходные данные:
        Словарь формата {"messages": [response]}, содержащая цепочку ответа на запрос
    """
    for msg in reversed(state["messages"]):
        if type(msg) is HumanMessage:
            break
    question = msg.content
    context = state["messages"][-1].content
    prompt = GENERATE_PROMPT.format(question=question, context=context)
    logging.debug(f"generate_answer called with prompt={prompt}")
    response = model_data.llm.invoke([{"role": "user", "content": prompt}])
    return {"messages": [response]}

def create_graph():
    """
    Создание графа обработки запроса через LLM.

    Граф выглядит так:
    START (начальный узел)
    |
    |
    generate_query_or_respond
    |                 |
    |                 | 
    тул не был вызван тул был вызван
    |                 |
    |                 |
    |                 retrieve (тул)
    |                 |
    |                 |
    |                 generate answer (на основе звпроса и результата работы тула)
    |                /
    |               /
    |---------------
    |
    V
    END

    Входные данные:
        Нет. Параметры задаются в model_data.set_parameters
    
    Выходные данные:
        Граф обработки запроса пользователя
    """
    workflow = StateGraph(MessagesState)

    # Define the nodes we will cycle between
    workflow.add_node(generate_query_or_respond)
    workflow.add_node("retrieve", ToolNode([model_data.retriever_tool]))
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

    checkpointer = InMemorySaver()
    return workflow.compile(checkpointer=checkpointer)

# Run with: uvicorn [filename]:app --reload