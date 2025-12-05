class ModelData:
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.llm = None
            self.retriever_tool = None
            self.top_doc_metadata = []
            self._initialized = True

    def set_parameters(self, llm, retriever_tool, vector_db, session_storage, graph):
        """
        Установка параметров графа - используемая LLM
        и тул для получения наиболее близких по смыслу
        текстов из БД
        """
        self.llm = llm
        self.retriever_tool = retriever_tool
        self.vector_db = vector_db
        self.session_storage = session_storage
        self.graph = graph

model_data = ModelData()
