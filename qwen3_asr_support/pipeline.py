import gc
from pathlib import Path

import numpy as np
import torch
from qwen3_asr_toolkit.audio_tools import WAV_SAMPLE_RATE, load_audio, process_vad
from silero_vad import load_silero_vad
from transformers import (
    AutoModelForMultimodalLM,
    AutoModelForTokenClassification,
    AutoProcessor,
)

from .model import (
    AlignedChunk,
    AlignedFragment,
    AlignedResult,
    AsrChunkScored,
    TokenAlternatives,
    TokenScored,
)


def model_to(model, device):
    if model is None:
        return model
    old_model = model
    res = old_model.to(device)
    del old_model
    gc.collect()
    torch.cuda.empty_cache()
    return res
    
ASR_LANGUAGE_TOKEN = 11528 # without preceding space, with preceding space is 4128.
ASR_START_TOKEN = 151704


def fragment_to_waveform(a):
    if isinstance(a, (str, Path)):
        return load_audio(str(a))
    elif isinstance(a, np.ndarray):
        return a
    else:
        raise TypeError(f"Unsupported type for audio {type(a)}")
        

class AlignedASR:
    def __init__(self, asr_model_id: str, aligner_model_id: str, local_files_only: bool=True, offload_immediately: bool = False, chunk: bool = True, align: bool = True):
        self._tokenizer_dictionary : list[str] | None = None
        self.asr_processor = AutoProcessor.from_pretrained(asr_model_id, local_files_only=local_files_only)
        self.asr_model = AutoModelForMultimodalLM.from_pretrained(asr_model_id, device_map="auto", local_files_only=local_files_only)

        self._align = align;
        
        if align:
            self.aligner_processor = AutoProcessor.from_pretrained(aligner_model_id, local_files_only=local_files_only)
            self.aligner_model = AutoModelForTokenClassification.from_pretrained(
                aligner_model_id, dtype=torch.bfloat16, device_map="auto", local_files_only=local_files_only
            )
        else:
            self.aligner_model = None

        if False:
            # Fails on:  Not enough SMs to use max_autotune_gemm mode
            self.asr_model = torch.compile(self.asr_model)
            self.aligner_model = torch.compile(self.aligner_model)
        
        self._offload_immediately = offload_immediately
        self._good_device = self.asr_model.device 
        if offload_immediately:
            # move them back to the cpu.
            self.asr_model = model_to(self.asr_model, "cpu")
            if self.aligner_model:
                self.aligner_model = model_to(self.aligner_model, "cpu")


        self._chunk = chunk

        if self._chunk:
            self._worker_vad_model = load_silero_vad(onnx=False)
            self._vad_segment_threshold = 120
            #wav_list = process_vad(wav, self._worker_vad_model, segment_threshold_s=_vad_segment_threshold)


    def models_to_cpu(self):
        self.aligner_model = model_to(self.aligner_model, "cpu")
        self.asr_model = model_to(self.asr_model, "cpu")


    def asr_chunk(self, audio_fragment, time_shift: float = 0.0,  language: str | None=None, align: bool = True) -> AlignedChunk:
        # Load model to GPU.
        self.asr_model = model_to(self.asr_model, self._good_device)

        # Step 1: Transcribe
        inputs = self.asr_processor.apply_transcription_request(audio=audio_fragment, language=language)
        inputs = inputs.to(self.asr_model.device, self.asr_model.dtype)
        with torch.inference_mode():
            output_ids = self.asr_model.generate(**inputs, max_new_tokens=256)
        generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
        parsed = self.asr_processor.decode(generated_ids, return_format="parsed")[0]
        transcript = parsed["transcription"]
        language = parsed["language"] or "English"
        
        if not self._align or not align:
            return AlignedChunk(fragments=[], language=language, transcript=transcript)

        
        if self._offload_immediately:
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

        if self._offload_immediately: 
            self.aligner_model = model_to(self.aligner_model, "cpu") 


        return AlignedChunk(fragments=[AlignedFragment(text = a["text"], start_time=a["start_time"]+time_shift, end_time=a["end_time"] + time_shift) for a in timestamps], language=language, transcript=transcript)


    def process(self, audio_url, label: str|None  = None,  language: str | None=None) -> AlignedResult:
        #audio_url = "https://huggingface.co/datasets/bezzam/audio_samples/resolve/main/librispeech_mr_quilter.wav"

        wav = fragment_to_waveform(audio_url)
    
        # Segment wav exceeding 3 minutes
        if len(wav) / WAV_SAMPLE_RATE >= 180 and self._chunk: 
            wav_list = process_vad(wav, self._worker_vad_model, segment_threshold_s=self._vad_segment_threshold)
        else:
            wav_list = [(0, len(wav), wav)]

        chunks = []
        for start_sample, end_sample, payload in wav_list:
            chunks.append(self.asr_chunk(payload, time_shift = start_sample / WAV_SAMPLE_RATE, language=language))
            
        if label is None and isinstance(audio_url, Path):
            label = audio_url.stem

        transcript = []
        fragments = []
        languages_found : list[str] = []
        for c in chunks:
            transcript.append(c.transcript)
            fragments.extend(c.fragments)
            if not c.language in languages_found:
                languages_found.append( c.language)

        transcript = " ".join(transcript)
        return AlignedResult(language=languages_found,transcript=transcript, label= label, fragments=fragments, chunks=chunks)



    def asr_chunk_scores(self, audio_fragment, topk=3, requested_tokens : list[int] | None = None,  language: str | None=None) -> AsrChunkScored:
        if isinstance(audio_fragment, list):
            wav_list = [fragment_to_waveform(z) for z in audio_fragment]
        else:
            wav_list = [fragment_to_waveform(audio_fragment)]

        # Load to the good device.
        self.asr_model = model_to(self.asr_model, self._good_device)
            
        inputs = self.asr_processor.apply_transcription_request(audio=wav_list, language=language)
        inputs = inputs.to(self.asr_model.device, self.asr_model.dtype)
        with torch.inference_mode(): 
            output_dict = self.asr_model.generate(**inputs, max_new_tokens=256,output_scores=True, return_dict_in_generate=True)
        
        # Offload back to the cpu.
        if self._offload_immediately:
            self.asr_model = model_to(self.asr_model, "cpu")

        output_ids = output_dict["sequences"]
        output_scores = output_dict["scores"]
        generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
        parsed = self.asr_processor.decode(generated_ids, return_format="parsed")[0]
        transcript = parsed["transcription"]
        language = parsed["language"] or "English"

   
        segments: list[TokenAlternatives] = []
        ranges = []
        position_this_far = None
        requested_score : list[TokenScored] | None = None if requested_tokens is None else []
        for i, s in enumerate(output_scores):
            alternatives : list[TokenScored] = []
            scores, indices = s.topk(topk)
            decoded = [self.asr_processor.tokenizer.decode([a]) for a in indices.tolist()[0]]

            if requested_score is not None and requested_tokens is not None:
                if i < len(requested_tokens):
                    # Append the score.
                    req_token = requested_tokens[i]
                    req_token_text = self.asr_processor.tokenizer.decode([req_token]) 
                    req_token_score = s[0, req_token]
                    requested_score.append(TokenScored(text=req_token_text, token=req_token, score=req_token_score))
                else:
                    requested_score.append(TokenScored(text="", token=0, score=0.0))
                    

            for token, score, text in zip(indices.tolist()[0], scores.tolist()[0], decoded):
                alternatives.append(TokenScored(text=text, score=score, token=token))
                
            segments.append(TokenAlternatives(alternatives=alternatives))
            if position_this_far is not None:
                ranges.append((position_this_far, position_this_far + len(alternatives[0].text)))
                position_this_far += len(alternatives[0].text)
            elif indices[0].tolist()[0]  == ASR_START_TOKEN:
                position_this_far = 0
        # requested_score
        return AsrChunkScored(segments=segments, transcript=transcript, language=language,ranges=ranges, requested_score=requested_score)

    def tokenizer_dictionary(self) -> list[str]:
        if self._tokenizer_dictionary is None:
            VOCAB_DICT_SIZE = 151936
            indices = range(VOCAB_DICT_SIZE + 1)
            self._tokenizer_dictionary =  [self.asr_processor.tokenizer.decode([a]) for a in indices]
        return self._tokenizer_dictionary

    def expected_tokens(self, text: str, language: str) -> list[int]:
        # Perform a longest prefix match on the dictionary.
        language = " " + language.strip().lower().capitalize()
        tokenizer_dictionary = self.tokenizer_dictionary()
        as_dict = {v: k for k,v in enumerate(tokenizer_dictionary)}
        language_token = as_dict[language]
        text_tokens = []
        remaining_text = text
        while remaining_text: 
            # Shortcut if we have a direct match.
            if remaining_text in as_dict:
                text_tokens.append(as_dict[remaining_text])
                break;

            best_str_token = None
            best_length = 0

            for tok, v in enumerate(tokenizer_dictionary):
                matching_start = remaining_text.startswith(v)
                if matching_start:
                    # See if it is a better fit.
                    if best_length < len(v):
                        best_str_token = tok
                        best_length = len(v)

            
            if not best_str_token:
                raise ValueError(f"cannot decompose text into expected tokens, failed to find prefix at {remaining_text}")
            remaining_text = remaining_text[best_length:]
            text_tokens.append(best_str_token)
            
        return [ASR_LANGUAGE_TOKEN, language_token,ASR_START_TOKEN] + text_tokens
