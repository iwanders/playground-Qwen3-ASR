from pydantic import BaseModel


class AlignedFragment(BaseModel):
    text: str
    start_time: float
    end_time: float

class AlignedChunk(BaseModel):
    transcript: str
    language: str
    fragments: list[AlignedFragment]

class AlignedResult(BaseModel):
    label: str | None
    transcript: str
    language: list[str]
    fragments: list[AlignedFragment]
    chunks: list[AlignedChunk]
