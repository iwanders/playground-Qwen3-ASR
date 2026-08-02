#!/usr/bin/env python3

import common  # noqa: F401, I001

import argparse 
import logging
from pathlib import Path

from qwen3_asr_support import AlignedASR

def run_asr_aligned(args):
    pipeline = AlignedASR(asr_model_id=args.asr_model, aligner_model_id=args.aligner_model, local_files_only=args.local_files_only)
    for f in args.files:
        r =  pipeline.process( f)
        print(r)
        output_dir = args.output_dir if args.output_dir else f.parent
        output_filename = f.stem
        output_dest = output_dir / f"{output_filename}.json"
        with open(output_dest, "w") as f:
            as_json = r.model_dump_json(indent=2)
            f.write(as_json)

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
    
    parser_run_asr_aligned.add_argument("--output-dir",  type=Path,  default=None, help="Output dir to write json files to, defaults to directory of input file." )
    
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
