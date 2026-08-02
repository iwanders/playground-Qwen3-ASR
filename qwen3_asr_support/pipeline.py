from pathlib import Path

import torch
from transformers import (
    AutoModelForMultimodalLM,
    AutoModelForTokenClassification,
    AutoProcessor,
)

from .model import AlignedFragment, AlignedResult


class AlignedASR:
    def __init__(self, asr_model_id: str, aligner_model_id: str, local_files_only: bool=True):
        self.asr_processor = AutoProcessor.from_pretrained(asr_model_id, local_files_only=local_files_only)
        self.asr_model = AutoModelForMultimodalLM.from_pretrained(asr_model_id, device_map="auto", local_files_only=local_files_only)
        
        self.aligner_processor = AutoProcessor.from_pretrained(aligner_model_id, local_files_only=local_files_only)
        self.aligner_model = AutoModelForTokenClassification.from_pretrained(
            aligner_model_id, dtype=torch.bfloat16, device_map="auto", local_files_only=local_files_only
        )

    def process(self, audio_url, label: str|None  = None) -> AlignedResult:
        #audio_url = "https://huggingface.co/datasets/bezzam/audio_samples/resolve/main/librispeech_mr_quilter.wav"
        
        # Step 1: Transcribe
        inputs = self.asr_processor.apply_transcription_request(audio=str(audio_url))
        inputs = inputs.to(self.asr_model.device, self.asr_model.dtype)
        output_ids = self.asr_model.generate(**inputs, max_new_tokens=256)
        generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
        parsed = self.asr_processor.decode(generated_ids, return_format="parsed")[0]
        transcript = parsed["transcription"]
        language = parsed["language"] or "English"

        
        # Step 2: Prepare alignment inputs
        aligner_inputs, word_lists = self.aligner_processor.prepare_forced_aligner_inputs(
            audio=str(audio_url), transcript=transcript, language=language,
        )
        aligner_inputs = aligner_inputs.to(self.aligner_model.device, self.aligner_model.dtype)
        
        # Step 3: Run forced aligner
        with torch.inference_mode():
            outputs = self.aligner_model(**aligner_inputs)
        
        # Step 4: Decode timestamps
        timestamps = self.aligner_processor.decode_forced_alignment(
            logits=outputs.logits,
            input_ids=aligner_inputs["input_ids"],
            word_lists=word_lists,
            timestamp_token_id=self.aligner_model.config.timestamp_token_id,
        )[0]

        if label is None and isinstance(audio_url, Path):
            label = audio_url.stem
            
 
        return AlignedResult(language=language,transcript=transcript, label= label, fragments=[AlignedFragment(**a) for a in timestamps])
