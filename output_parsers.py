from typing import List, Dict, Any
from langchain_core.output_parsers import PydanticOutputParser

from pydantic import BaseModel, Field

class Response(BaseModel):
    truth: List[str] = Field(description="any truthful information about the topic ")
    lie: List[str] = Field(description="any misleading informatin the topic")
    banter: List[str] = Field(description="any part of the response not directly pertaining to the topic")

    def to_dict(self) -> Dict[str, Any]:
        return {"truth": self.truth, "lie": self.lie, "banter": self.banter}
    
response_parser = PydanticOutputParser(pydantic_object=Response)
