from langchain_chroma import Chroma

def load_vector_store(collection_name: str, embeddings):
    """
    Загрузки векторной БД. Если она ещё не была создана,
    её необходимо создать при помощи db_creator.py

    Входные данные:
        collection_name - имя коллекции в БД
        embeddings - использованные эмбеддинги при создании БД

    Выходные данные:
        Векторная БД
    """

    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not necessary
    )

    if len(vector_store.get()['ids']) == 0:
        print("База данных пуста. Необходимо создать БД при помощи db_creator.py")
        raise ValueError

    return vector_store