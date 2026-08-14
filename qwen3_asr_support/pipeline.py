import gc
from pathlib import Path

import torch
from qwen3_asr_toolkit.audio_tools import WAV_SAMPLE_RATE, load_audio, process_vad
from silero_vad import load_silero_vad
from transformers import (
    AutoModelForMultimodalLM,
    AutoModelForTokenClassification,
    AutoProcessor,
)

from .model import AlignedChunk, AlignedFragment, AlignedResult, TokenScored, AsrChunkScored, TokenAlternatives


def model_to(model, device):
    old_model = model
    res = old_model.to(device)
    del old_model
    gc.collect()
    torch.cuda.empty_cache()
    return res
    


class AlignedASR:
    def __init__(self, asr_model_id: str, aligner_model_id: str, local_files_only: bool=True, shuffle_memory: bool = False, chunk: bool = True):
        self.asr_processor = AutoProcessor.from_pretrained(asr_model_id, local_files_only=local_files_only)
        self.asr_model = AutoModelForMultimodalLM.from_pretrained(asr_model_id, device_map="auto", local_files_only=local_files_only)
        
        self.aligner_processor = AutoProcessor.from_pretrained(aligner_model_id, local_files_only=local_files_only)
        self.aligner_model = AutoModelForTokenClassification.from_pretrained(
            aligner_model_id, dtype=torch.bfloat16, device_map="auto", local_files_only=local_files_only
        )

        if False:
            # Fails on:  Not enough SMs to use max_autotune_gemm mode
            self.asr_model = torch.compile(self.asr_model)
            self.aligner_model = torch.compile(self.aligner_model)
        
        self._shuffle_memory = shuffle_memory
        if shuffle_memory:
            self._good_device = self.asr_model.device 
            # move them back to the cpu.
            self.aligner_model = model_to(self.aligner_model, "cpu")
            self.asr_model = model_to(self.asr_model, "cpu")


        self._chunk = chunk

        if self._chunk:
            self._worker_vad_model = load_silero_vad(onnx=False)
            self._vad_segment_threshold = 120
            #wav_list = process_vad(wav, self._worker_vad_model, segment_threshold_s=_vad_segment_threshold)
            

    def asr_chunk(self, audio_fragment, time_shift: float = 0.0) -> AlignedChunk:
    
        if self._shuffle_memory:
            self.asr_model = model_to(self.asr_model, self._good_device)
            
        
        # Step 1: Transcribe
        inputs = self.asr_processor.apply_transcription_request(audio=audio_fragment)
        inputs = inputs.to(self.asr_model.device, self.asr_model.dtype)
        with torch.inference_mode():
            output_ids = self.asr_model.generate(**inputs, max_new_tokens=256)
        generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
        parsed = self.asr_processor.decode(generated_ids, return_format="parsed")[0]
        transcript = parsed["transcription"]
        language = parsed["language"] or "English"

        
        if self._shuffle_memory:
            # Move it back to the cpu.
            self.asr_model = model_to(self.asr_model, "cpu") 
            
            # Move the aligner model to the good device.
            self.aligner_model = model_to(self.aligner_model, self._good_device) 
            
        # Step 2: Prepare alignment inputs
        aligner_inputs, word_lists = self.aligner_processor.prepare_forced_aligner_inputs(
            audio=audio_fragment, transcript=transcript, language=language,
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

        if self._shuffle_memory: 
            self.aligner_model = model_to(self.aligner_model, "cpu") 

        return AlignedChunk(fragments=[AlignedFragment(text = a["text"], start_time=a["start_time"]+time_shift, end_time=a["end_time"] + time_shift) for a in timestamps], language=language, transcript=transcript)


    def process(self, audio_url, label: str|None  = None) -> AlignedResult:
        #audio_url = "https://huggingface.co/datasets/bezzam/audio_samples/resolve/main/librispeech_mr_quilter.wav"

        wav = load_audio(str(audio_url))
    
        # Segment wav exceeding 3 minutes
        if len(wav) / WAV_SAMPLE_RATE >= 180 and self._chunk: 
            wav_list = process_vad(wav, self._worker_vad_model, segment_threshold_s=self._vad_segment_threshold)
        else:
            wav_list = [(0, len(wav), wav)]

        chunks = []
        for start_sample, end_sample, payload in wav_list:
            chunks.append(self.asr_chunk(payload, time_shift = start_sample / WAV_SAMPLE_RATE))
            
        if label is None and isinstance(audio_url, Path):
            label = audio_url.stem

        transcript = []
        fragments = []
        language = []
        for c in chunks:
            transcript.append(c.transcript)
            fragments.extend(c.fragments)
            if not c.language in language:
                language.append( c.language)

        transcript = " ".join(transcript)
        return AlignedResult(language=language,transcript=transcript, label= label, fragments=fragments, chunks=chunks)



    def asr_chunk_scores(self, audio_url, topk=3) -> AsrChunkScored:
        wav = load_audio(str(audio_url))
        wav_list = [wav]
        if self._shuffle_memory:
            self.asr_model = model_to(self.asr_model, self._good_device)
            
         
        inputs = self.asr_processor.apply_transcription_request(audio=wav_list)
        inputs = inputs.to(self.asr_model.device, self.asr_model.dtype)
        with torch.inference_mode(): 
            output_dict = self.asr_model.generate(**inputs, max_new_tokens=256,output_scores=True, return_dict_in_generate=True)
        
        output_ids = output_dict["sequences"]
        output_scores = output_dict["scores"]
        generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
        parsed = self.asr_processor.decode(generated_ids, return_format="parsed")[0]
        transcript = parsed["transcription"]
        language = parsed["language"] or "English"

   
        segments: list[TokenAlternatives] = []
        for i, s in enumerate(output_scores):
            alternatives : list[TokenScored] = []
            scores, indices = s.topk(topk)
            decoded = [self.asr_processor.tokenizer.decode([a]) for a in indices.tolist()[0]]
            
            for token, score, text in zip(indices.tolist()[0], scores.tolist()[0], decoded):
                
                alternatives.append(TokenScored(text=text, score=score, token=token))
                
            segments.append(TokenAlternatives(alternatives=alternatives))
            
        return AsrChunkScored(segments=segments, transcript=transcript, language=language)
