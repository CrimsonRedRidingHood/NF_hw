from dotenv import load_dotenv
load_dotenv()

from langchain_ollama import OllamaEmbeddings

USE_LOCAL_MODEL = True
USED_MODEL = "GPT-4o"

COLLECTION_NAME = "nf_hw_collection"

embeddings = OllamaEmbeddings(
    model="qwen3-embedding",
)