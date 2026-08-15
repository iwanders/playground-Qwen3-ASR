# Playground Qwen3-ASR

Some wrapper tooling around qwen3-asr, including the forced aligner to make timestamped transcripts.

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

License is Apache, same as qwen3-asr.
