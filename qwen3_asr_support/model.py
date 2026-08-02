from pydantic import BaseModel


class AlignedFragment(BaseModel):
    text: str
    start_time: float
    end_time: float

class AlignedResult(BaseModel):
    label: str | None
    transcript: str
    language: str
    fragments: list[AlignedFragment]
