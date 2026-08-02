#!/usr/bin/env python3

import common  # noqa: F401, I001

import argparse
import logging
import torch
from transformers import AutoProcessor, AutoModelForMultimodalLM, AutoModelForTokenClassification
from pathlib import Path
from pydantic import BaseModel

class Aligned(BaseModel):
    text: str
    start_time: float
    end_time: float

class AlignedASR:
    def __init__(self, asr_model_id: str, aligner_model_id: str, local_files_only: bool=True):
        self.asr_processor = AutoProcessor.from_pretrained(asr_model_id, local_files_only=local_files_only)
        self.asr_model = AutoModelForMultimodalLM.from_pretrained(asr_model_id, device_map="auto", local_files_only=local_files_only)
        
        self.aligner_processor = AutoProcessor.from_pretrained(aligner_model_id, local_files_only=local_files_only)
        self.aligner_model = AutoModelForTokenClassification.from_pretrained(
            aligner_model_id, dtype=torch.bfloat16, device_map="auto", local_files_only=local_files_only
        )

    def process(self, audio_url):
        #audio_url = "https://huggingface.co/datasets/bezzam/audio_samples/resolve/main/librispeech_mr_quilter.wav"
        
        # Step 1: Transcribe
        inputs = self.asr_processor.apply_transcription_request(audio=audio_url)
        inputs = inputs.to(self.asr_model.device, self.asr_model.dtype)
        output_ids = self.asr_model.generate(**inputs, max_new_tokens=256)
        generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
        parsed = self.asr_processor.decode(generated_ids, return_format="parsed")[0]
        transcript = parsed["transcription"]
        language = parsed["language"] or "English"
        
        # Step 2: Prepare alignment inputs
        aligner_inputs, word_lists = self.aligner_processor.prepare_forced_aligner_inputs(
            audio=audio_url, transcript=transcript, language=language,
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

        return [Aligned(**a) for a in timestamps]

 


def run_asr_aligned(args):
    pipeline = AlignedASR(asr_model_id=args.asr_model, aligner_model_id=args.aligner_model, local_files_only=args.local_files_only)
    for f in args.files:
        r =  pipeline.process(str(f))
        print(r)

if __name__ == "__main__":
    # Create a parser with some subcommands
    parser = argparse.ArgumentParser(description="asr thingy")
    _ = parser.add_argument(
        "-v",
        "--verbose",
        help="Enable verbose output",
        action="store_true",
        default=False,
    ) 
    # Add subcommands
    subparsers = parser.add_subparsers(dest="command", help="sub-command help")
    
    asr_model_id = "Qwen/Qwen3-ASR-0.6B-hf"
    aligner_model_id = "Qwen/Qwen3-ForcedAligner-0.6B-hf"
    parser_run_asr_aligned = subparsers.add_parser("asr_aligned", help="run asr with alignment")
    parser_run_asr_aligned.add_argument("--allow-download", dest="local_files_only", default=True, action="store_false")
    parser_run_asr_aligned.add_argument("--asr-model", type=str, help="The asr model to use %(default)s", default=asr_model_id)
    parser_run_asr_aligned.add_argument("--aligner-model", type=str, help="The aligner model to use %(default)s", default=aligner_model_id)
    parser_run_asr_aligned.add_argument("files",
        nargs='+',
         type=Path,
         help="Paths to operate on, retrieve https://huggingface.co/datasets/bezzam/audio_samples/resolve/main/librispeech_mr_quilter.wav as an example",
    )
    
    parser_run_asr_aligned.set_defaults(func=run_asr_aligned)
    
    
 
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    # Execute the selected command's function
    if args.command:
        args.func(args)
    else:
        parser.print_help()
