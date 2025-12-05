from langchain_classic.tools.retriever import create_retriever_tool
from langchain_community.tools import DuckDuckGoSearchResults
from state import model_data
from langchain_core.tools import tool
from typing import Any, Optional, Dict

search = DuckDuckGoSearchResults()

@tool(description="Возвращает ближайшие по смыслу записи из базы")
def retrieval_function(x: Any, k: int = 5, snippet_len: int = 30, filter: Optional[Dict[str, str]] = None, **kwargs):
    retval = model_data.vector_db.similarity_search(x, k, filter, **kwargs)
    model_data.top_doc_metadata = [
        {
            "source": d.metadata["source"],
            "snippet": d.page_content[:snippet_len] + ("..." if len(d.page_content) > snippet_len else "")
        }
        for d in retval
    ]
    return retval

retriever_tool = create_retriever_tool(
    retrieval_function,
    "retrieve_neoflex_info",
    "Найти и вернуть информацию об организации 'Неофлекс'."
)
