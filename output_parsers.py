from typing import List, Dict, Any
from langchain_core.output_parsers import PydanticOutputParser

from pydantic import BaseModel, Field

class Response(BaseModel):
    truth: List[str] = Field(description="any truthful information about the topic ")
    lie: List[str] = Field(description="any misleading informatin the topic")
    banter: List[str] = Field(description="any part of the response not directly pertaining to the topic")

    def to_dict(self) -> Dict[str, Any]:
        return {"truth": self.truth, "lie": self.lie, "banter": self.banter}
    
class StartupResponse(BaseModel):
    name: str = Field(description="the name of the stratup that raised funding")
    industry: str = Field(description="their geographic location")
    website: str = Field(description="a link to their website url")
    amount_raised: str = Field(description="the amount raised and indicatation of whether it was cash or euity (best guess)")
    date_raised: str = Field(description="the approximate date when the funds were raised (best guess if you don't know the exact date)")
    location: str = Field(description="the geographic location of the startup")
    stage_of_funding: str = Field(description="the funding stage like Seed / Series A / Series B etc")
    validaton: str = Field(description="why you think they raised funding. ie a link to the announcement or chain of thought reasoning leading to that conclusion")

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "industry": self.industry, "website": self.website, "amount_raised": self.amount_raised, "date_raised": self.date_raised, "location": self.location, "stage_of_funding": self.stage_of_funding, "validaton": self.validaton}


response_parser = PydanticOutputParser(pydantic_object=Response)

startup_parser = PydanticOutputParser(pydantic_object=StartupResponse)
