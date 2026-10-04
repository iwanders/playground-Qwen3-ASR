# Qwen3-ASR server

Some wrapper tooling around qwen3-asr, including the forced aligner to make timestamped transcripts.

The `qwen3_asr_support` module provides a wrapper class `AlignedASR` class that wraps the actual model in helper tooling.
The structs in `model.py` define the interaction and returns from that class.
The `PipelineWorker` class provides a threaded worker that runs the actual ASR and can be integrated into async servers, an example is provided in `example_server`.

The `asr_chunk_scores` method outputs the top 'n' tokens for each position and can provide the score against an expected token sequence.

Note that to access the microphone on iOS requires https.

## CLI
```
./main.py  asr_aligned /tmp/our_audio_with_voice.mp3  --output-dir /tmp/foobar/
# writes /tmp/foobar/our_audio_with_voice.json
```

Json file structured like:
```
{
    "label": "our_audio_with_voice",
    "transcript": "Full text of all words",
    "language": [
        "English"
    ],
    "fragments": [
        {
            "text": "Full",
            "start_time": 1.84,
            "end_time": 2.16
        }
    ],
    "chunks": [
     {
       "transcript": "Full text of all words",
       "language": "English",
       "fragments": [{
            "text": "Full",
            "start_time": 1.84,
            "end_time": 2.16
        }]
    }
}
```

## HTTP server
Mocked up a webpage & server combination that transcribes voice segments.
```
python3 -m streaming_asr.server server
```

or for iOS, access to the microphone needs ssl:
```
python3 -m streaming_asr.server server --ssl
```

The model itself expects a sample rate of 16000 Hz, silero vad downsamples to that, but if you create your smaples in
another way be sure to account for that.

## Aligned Viewer

There's a viewer for the aligned files:
```
python3 -m aligned_viewer.server server /tmp/sdfdsf/
```
Directory expects mp3 files with their json files (same basename). Json files are produced with:
```
./main.py asr_aligned --force-vad --output-dir /tmp/sdfdsf/ /tmp/sdfdsf/*.mp3
```



## Notes

Some issues reported upstream with timestamps, see <a href="https://github.com/QwenLM/Qwen3-ASR/issues/197">this</a> issue.
There's a <a href="https://github.com/QwenLM/Qwen3-ASR/issues/197#issuecomment-5965450292">solution</a> proposed there that we may want to implement.


## License
License is Apache, same as qwen3-asr.
