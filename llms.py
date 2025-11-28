from langchain_openai import ChatOpenAI
import os
import getpass
from langchain_ollama import ChatOllama

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

def create_llm(use_local_model: bool, used_model_name: str=""):
    """
    Создание LLM-модели по заданным параметрам, см. "входные данные"

    Входные данные:
        use_local_model: bool - если этот параметр выставлен в True,
        используется локально развёрнутая, через Ollama, модель. Имя
        конкретной модели на данный момент захардкожено. Если этот
        параметр выставлен в False, создаётся модель через ChatOpenAI
        (openrouter), имя которой хранится в аргументе used_model_name.

        used_model_name: str - имя модели на OpenRouter.ai. Используется,
        если параметр use_local_model имеет значение False.
    
    Выходные данные:
        Модель для использования в графе. Либо ChatOllama, либо ChatOpenAI,
        в зависимости от значения параметра use_local_model (True, False
        соответственно).
    """
    if use_local_model:
        return ChatOllama(
            model="gpt-oss:20b",
            temperature=0,
        )
    if not os.environ.get("OPENROUTER_API_KEY"):
        os.environ["OPENROUTER_API_KEY"] = getpass.getpass("Enter API key for OpenRouter: ")

    return ChatOpenAI(
        max_tokens=model_data[used_model_name]["tokens"],
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url="https://openrouter.ai/api/v1",
        model=model_data[used_model_name]["name"],
        default_headers={
            "HTTP-Referer": "YOUR_SITE_URL", # Optional. Site URL for rankings on openrouter.ai.
            "X-Title": "YOUR_SITE_NAME", # Optional. Site title for rankings on openrouter.ai.
        }
    )