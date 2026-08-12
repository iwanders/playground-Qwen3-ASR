#!/usr/bin/env python3

import common  # noqa: F401, I001
import types
import sys
from dataclasses import dataclass



from transformers import AutoModelForMultimodalLM, AutoProcessor  # noqa: I001

use_file = "../librispeech_mr_quilter.wav"
if len(sys.argv) > 1:
    use_file = sys.argv[1]

# https://huggingface.co/docs/transformers/v5.14.0/en/model_doc/qwen3_asr#transformers.Qwen3ASRProcessor
model_id = "Qwen/Qwen3-ASR-0.6B-hf"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForMultimodalLM.from_pretrained(model_id, device_map="auto")
print(f"Model loaded on {model.device} with dtype {model.dtype}")

inputs = processor.apply_transcription_request(
    # audio="https://huggingface.co/datasets/bezzam/audio_samples/resolve/main/librispeech_mr_quilter.wav",
    # audio="../librispeech_mr_quilter.wav",
    audio=use_file,
).to(model.device, model.dtype)
print(inputs)
print(f'input_ids shape: {inputs["input_ids"].shape}')
inputs_entries_before = inputs["input_ids"].size(1)
print(f'input_features shape: {inputs["input_features"].shape}')
output_dict = model.generate(**inputs, max_new_tokens=256,output_scores=True, return_dict_in_generate=True)
print(f"output_dict: {output_dict}")
output_ids = output_dict["sequences"]
print(f"sequences shape: {output_ids.shape}")
output_entries_after = output_ids.size(1)

VOCAB_SIZE = 151936

output_scores = output_dict["scores"]

generated_outputs = output_entries_after - inputs_entries_before
print(f"generated_outputs; {generated_outputs}")
assert generated_outputs == len(output_scores)

@dataclass
class TokenAlternatives:
    indices: list[int]
    scores: list[float] 

highest_score_index: list[TokenAlternatives] = []
print(f"scores shape tuple of; {len(output_scores)}")
for i, s in enumerate(output_scores):
    #print(f"shape of score tuple {i}; {s.shape}")
    assert s.numel() == VOCAB_SIZE
    scores, indices = s.topk(3)
    r = TokenAlternatives(scores=scores.tolist()[0], indices=indices.tolist()[0])
    highest_score_index.append(r)
    


# Next, lets check if the output ids actually match the peak in the scores...


    
generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
print("output_ids:", output_ids)
print(f"highest_score_index; {highest_score_index}")
print("generated:", generated_ids)
# Raw output includes language tag and <asr_text> marker
raw = processor.decode(generated_ids)[0]
print(f"Raw: {raw}")





# Parsed output: dict with "language" and "transcription"
def monkeypatched_decode(self,  *args,return_format="raw",  **kwargs):
     #valid_formats = ["raw", "parsed", "transcription_only"]
    #if return_format not in valid_formats:
    #    raise ValueError(f"return_format must be one of {valid_formats}.")
    if return_format != "raw":
        kwargs["skip_special_tokens"] = True

    ids = args[0].tolist()[0]
    print("ids:", ids)
    # ids_to_tokens = self.tokenizer.convert_tokens_to_string(ids )
    segments = [self.tokenizer.decode([i]) for i in ids]
    print("segments:", segments)
    print("segments joined :", repr("".join(segments)))
    
    #self.convert_tokens_to_string(self.convert_ids_to_tokens(token_ids))
    decoded = self.tokenizer.decode(*args, **kwargs) 
    if return_format == "parsed":
        decoded = self.parse_output(decoded)
    elif return_format == "transcription_only":
        decoded = self.extract_transcription(decoded)
    return decoded 
processor.decode = types.MethodType(monkeypatched_decode, processor)

parsed = processor.decode(generated_ids, return_offsets_mapping=True, return_format =None) 
print(f"Parsed: {parsed}")

# Extract only the transcription text
transcription = processor.decode(generated_ids, return_format="transcription_only")[0]
print(f"Transcription: {transcription}")


# Retrieve the alternatives:
print(generated_ids)
for a in highest_score_index:
    for score, index in zip(a.scores, a.indices):
        print(f"{score} with {index}")
        string = processor.tokenizer.decode([ index]) 
        print(f"     {score:2.2f}: {string}")
    print()

"""
Raw: language English<asr_text>Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel.
Parsed: {'language': 'English', 'transcription': 'Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel.'}
Transcription: Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel.
"""
