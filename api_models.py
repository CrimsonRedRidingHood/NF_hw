from typing import List
from pydantic import BaseModel

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