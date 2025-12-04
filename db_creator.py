from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
import json
import sys

if len(sys.argv) != 4:
    print("Usage: python db_creator.py [JSON file with data] [db path] [collection name]")
    quit()

docs_filename = sys.argv[1]
db_path = sys.argv[2]
collection_name = sys.argv[3]

print('Remember that embeddings in the db_creator and those in the main app should be the same!')
print('Are you ready to proceed? (Y/y for positive answer, anything else for negative)')

if input().lower() != 'y':
    quit()

embeddings = OllamaEmbeddings(
    model="qwen3-embedding",
)

vector_store = Chroma(
    collection_name=collection_name,
    embedding_function=embeddings,
    persist_directory=db_path,  # Where to save data locally, remove if not necessary
)

vector_store.reset_collection()

with open(docs_filename, 'r', encoding='utf-8') as file:
    docs_from_file = json.load(file)

splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", ".", " "],
    chunk_size=1000,
    chunk_overlap=100,
)

documents = splitter.create_documents(
    [obj["text"] for obj in docs_from_file],
    metadatas=[{"source": obj["url"], "section": obj["section"]} for obj in docs_from_file],
)

vector_store.add_documents(documents=documents)