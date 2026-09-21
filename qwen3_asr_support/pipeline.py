import gc
import hashlib
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
        """Automatic Speech Recognition with optional forced alignment.
    
        This class provides ASR capabilities using Qwen3-ASR models, with optional
        word-level timestamp alignment using Qwen3-ForcedAligner.
    
        Parameters
        ----------
        asr_model_id : str
            The Hugging Face model ID for the ASR (Automatic Speech Recognition) model.
            Used to load both the ASR processor and model.
            asr_model_id = "Qwen/Qwen3-ASR-0.6B-hf"
            or
            asr_model_id = "Qwen/Qwen3-ASR-1.7B-hf"
        aligner_model_id : str
            The Hugging Face model ID for the forced aligner model. Used when
            `align=True` to compute word-level timestamps on the transcript.
            aligner_model_id = "Qwen/Qwen3-ForcedAligner-0.6B-hf"
        local_files_only : bool, optional
            If True, models will be loaded only from local cache. If False and models
            aren't cached, they will be downloaded from Hugging Face. Default is True.
        offload_immediately : bool, optional
            If True, moves ASR and aligner models to CPU immediately after initialization
            to conserve VRAM. Models are lazily loaded to GPU on first use. Default is False.
        chunk : bool, optional
            If True, enables chunk-based processing for audio exceeding 3 minutes using
            VAD (Voice Activity Detection) segmentation. Default is True.
        align : bool, optional
            If True, enables forced alignment which computes word-level start/end timestamps
            for each transcribed word. If False, the aligner model is not loaded.
            Default is True.
        """
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
        """
            Force models to the cpu immediately.
        """
        self.aligner_model = model_to(self.aligner_model, "cpu")
        self.asr_model = model_to(self.asr_model, "cpu")


    def asr_chunk(self, audio_fragment, time_shift: float = 0.0,  language: str | None=None, align: bool = True) -> AlignedChunk:
        """
            Run ASR on a single audio fragment and output an aligned chunk.
            
            Parameters
            ----------
            audio_fragment : np.ndarray
                Audio samples to run ASR on.
                
            time_shift : float
                Timeshift value to apply to the aligned chunks, this makes working with sliced audio fragments easier.
                
            language : str | None, optional
                Language to pass to the transcription request, this does not mean that the return chunk is always in
                this language.
                
            align : bool, optional
                Wether or not to populate the fragments with their individual alignment values, if false the alignment
                is skipped.
                
        """
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


    def process(self, audio_url, label: str|None  = None,  language: str | None=None, force_vad: bool = False, checkpoint_dir: Path| None = None) -> AlignedResult:
        """
            Process the provided audio url or waveform.

            This runs a VAD model under the hood to split long audio into < 180 s intervals.
            
            Parameters
            ----------
            label : str|None 
                The label to propagate to the AlignedResult, set to the stem of the audio url if it is an url.
                
            time_shift : float
                Timeshift value to apply to the aligned chunks, this makes working with sliced audio fragments easier.
                
            language : str | None, optional
                Passed to asr_chunk.
                
            force_vad : bool
                If true, VAD is forced to run even if self._chunk is false or if the sample is less than 3 minutes.

            checkpoint_dir : Path | None
                The path to write the checkpoints to if any, this is keyed by audio only!
        """
        
        #audio_url = "https://huggingface.co/datasets/bezzam/audio_samples/resolve/main/librispeech_mr_quilter.wav"

        wav = fragment_to_waveform(audio_url)
        sha_hash = hashlib.sha256(wav.tobytes()).hexdigest()

    
        # Segment wav exceeding 3 minutes
        if len(wav) / WAV_SAMPLE_RATE >= 180 and self._chunk or force_vad: 
            wav_list = process_vad(wav, self._worker_vad_model, segment_threshold_s=self._vad_segment_threshold)
        else:
            wav_list = [(0, len(wav), wav)]

        if label is None and isinstance(audio_url, Path):
            label = audio_url.stem

        result = AlignedResult(language = [], transcript="", label=label, fragments = [], chunks = [])
        filepath = None
        if checkpoint_dir:
            filepath = checkpoint_dir / f"{label}_{sha_hash}.json"
            if filepath.is_file():
                with filepath.open("r") as f:
                    result = AlignedResult.model_validate_json(f.read())
        
        def flush_result():
            if filepath:
                with filepath.open("w") as f:
                    f.write(result.model_dump_json(indent=2, ensure_ascii=False))
            
 
        languages_found : list[str] = []
        for windex in range(len(result.chunks), len(wav_list)):
            start_sample, end_sample, payload = wav_list[windex]
            c = self.asr_chunk(payload, time_shift = start_sample / WAV_SAMPLE_RATE, language=language)
            result.fragments.extend(c.fragments)
            if not c.language in languages_found:
                languages_found.append( c.language)
            result.chunks.append(c)
            flush_result()
            

        transcript = " ".join(c.transcript for c in result.chunks)
        result.transcript = transcript
        return result



    def asr_chunk_scores(self, audio_fragment: str | np.ndarray, topk: int=3, requested_tokens : list[int] | None = None,  language: str | None=None) -> AsrChunkScored:
        """
            Process a chunk and score its output against the requested tokens, also returns the 'topk' tokens for each position.

            This can be used to score how well a particular audio segment pronounced individual tokens.
            
            Parameters
            ----------
            audio_fragment : str | np.ndarray
                The audio fragment, either an URL or a 16000 Hz waveform.
                
            topk : int
                The top 'k' elements to return for each token identified.
                
            requested_tokens : list[int] | None, optional
                The requested / expected tokens against which to calculate the score.  
                
            language : str | None, optional
                Passed to asr_chunk.                   
        """
        
        
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
        """
            Emits the entire tokenizer dictionary as a list of token values. The token's index is its integer value.             
        """
        
        if self._tokenizer_dictionary is None:
            VOCAB_DICT_SIZE = 151936
            indices = range(VOCAB_DICT_SIZE + 1)
            self._tokenizer_dictionary =  [self.asr_processor.tokenizer.decode([a]) for a in indices]
        return self._tokenizer_dictionary

    def expected_tokens(self, text: str, language: str) -> list[int]:
        """
            Use the tokenizer to encode the provided text into a sequence of tokens, including the language prefix.

            This is not tested that thoroughly, but seems to work for the particular use cases I'm intested in.
 
            Parameters
            ----------
            text : str
                The text to tokenize.

            language : str
                Language token to use.              
        """
        # Perform a longest prefix match on the dictionary.
        language = " " + language.strip().lower().capitalize()
        tokenizer_dictionary = self.tokenizer_dictionary()
        as_dict = {v: k for k,v in enumerate(tokenizer_dictionary)}
        language_token = as_dict[language]
        
        def tokenize_prefix_matcher(text: str):  # pyright: ignore[reportUnusedFunction]
            """
            Try to create a list of expected tokens for a provided string.

            This performs a longest-prefix match approach to convert text to tokens, taken the token that's the longest
            matching one for each word or sequence of words.

            """
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
                    if matching_start and best_length < len(v):
                            best_str_token = tok
                            best_length = len(v)
    
                
                if not best_str_token:
                    raise ValueError(f"cannot decompose text into expected tokens, failed to find prefix at {remaining_text}")
                remaining_text = remaining_text[best_length:]
                text_tokens.append(best_str_token)
            return text_tokens

        text_tokens = self.asr_processor.tokenizer.encode(text)
        return [ASR_LANGUAGE_TOKEN, language_token,ASR_START_TOKEN] + text_tokens
