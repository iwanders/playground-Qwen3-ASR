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

class TokenScored(BaseModel):
    text: str
    token: int
    score: float


class TokenAlternatives(BaseModel):
    alternatives: list[TokenScored]

class AsrChunkScored(BaseModel):
    transcript: str
    language: str
    segments: list[TokenAlternatives]
    
    # Ranges for each segment.
    ranges: list[tuple[int,int]]

    # Score for the requested token sequence.
    requested_score: list[TokenScored] | None = None
