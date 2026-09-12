# Qwen3-ASR server

Some wrapper tooling around qwen3-asr, including the forced aligner to make timestamped transcripts.

The `qwen3_asr_support` module provides a wrapper class `AlignedASR` class that wraps the actual model in helper tooling.
The structs in `model.py` define the interaction and returns from that class.
The `PipelineWorker` class provides a threaded worker that runs the actual ASR and can be integrated into async servers, an example is provided in `example_server`.

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




## License
License is Apache, same as qwen3-asr.
