#!/usr/bin/env python3
import common  # noqa: I001


import numpy as np 
from aiohttp import web
import aiohttp
import torchaudio
from pathlib import Path
import argparse
from qwen3_asr_support.pipeline import AlignedASR

from torchcodec.decoders import AudioDecoder
import asyncio

import io
from qwen3_asr_support.pipeline_worker import TestAbstraction, PipelineWorker, PipelineAbstraction, AsyncTask, TaskType

THIS_PATH = Path(__file__).parent.absolute()




async def handle(request):
    return web.FileResponse(THIS_PATH/'index.html')

    
async def websocket_handler(request):
    pipeline_worker = request.app['pipeline']

    ws = web.WebSocketResponse()
    await ws.prepare(request)
    loop = asyncio.get_running_loop()


    async for msg in ws: 
        if msg.type == aiohttp.WSMsgType.TEXT:
            if msg.data == 'close':
                await ws.close()
            else:
                await ws.send_str(msg.data + '/answer')
        elif msg.type == aiohttp.WSMsgType.ERROR:
            print('ws connection closed with exception %s' %
                  ws.exception())
        elif msg.type == aiohttp.WSMsgType.BINARY:
            #waveform, sample_rate = torchaudio.load(msg.data)
            #audio_handler.write(msg.data) 
            #for waveform in decoder: 
            #    print(f"Got waveform that was {len(waveform)} long at  Hz.")
            print(f"Got packed of {len(msg.data)} long")
            task_type = TaskType.ASR_CHUNK
            # asr_chunk(self, audio_fragment, time_shift: float = 0.0)
             
          
            audio_waveform = np.frombuffer(msg.data, dtype=np.float32) 
            payload = {
                "audio_fragment": [audio_waveform],
            }
            task = AsyncTask(type=task_type, payload=payload)
            def pipeline_entry(data):
                f = pipeline_worker.enqueue(data)
                try: 
                    r = f.result().model_dump_json()
                except Exception as e:
                    r = str(e)
                return r
            result = await loop.run_in_executor(None, pipeline_entry, task)
            await ws.send_str(result)
            # This is a float32 array.
            # Cool, convert this into ehm, the actual samples, and dispatch to the pipeline?
             
    print('websocket connection closed')

    return ws

app = web.Application()
app.add_routes([web.get('/', handle),
                web.get('/{name}', handle), web.get('/ws', websocket_handler)])

def run_server(args):
    pipeline = AlignedASR(asr_model_id=args.asr_model, aligner_model_id=args.aligner_model, local_files_only=args.local_files_only, shuffle_memory=args.reduce_memory)
    work_abstraction = PipelineAbstraction(pipeline)
    pipeline_worker = PipelineWorker(work_abstraction)
    app["pipeline"] = pipeline_worker

    ssl_context = None
    if args.ssl:
        import ssl
        ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        
        # Load your certificate and private key files
        ssl_context.load_cert_chain(certfile=THIS_PATH/"certs"/'cert.pem', keyfile=THIS_PATH/"certs"/'key.pem')

    port = args.port
    if ssl_context is not None:
        port = args.ssl_port

    web.run_app(app, port=port, ssl_context=ssl_context)


if __name__ == '__main__':
    # Create a parser with some subcommands
    parser = argparse.ArgumentParser(description="asr server")
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
    parser_run_asr_aligned = subparsers.add_parser("server", help="run asr with alignment")
    parser_run_asr_aligned.add_argument("--allow-download", dest="local_files_only", default=True, action="store_false")
    parser_run_asr_aligned.add_argument("--reduce-memory", default=False, action="store_true")
    parser_run_asr_aligned.add_argument("--asr-model", type=str, help="The asr model to use %(default)s", default=asr_model_id)
    parser_run_asr_aligned.add_argument("--aligner-model", type=str, help="The aligner model to use %(default)s", default=aligner_model_id)
    parser_run_asr_aligned.add_argument("--ssl",  default=False, action="store_true", help="use ssl, for ios to use mic")
    parser_run_asr_aligned.add_argument("--ssl-port",  type=int,  default=8001, help="Port to bind to when using ssl" )
    
    parser_run_asr_aligned.add_argument("--port",  type=int,  default=8000, help="Port to bind to." )
    
    parser_run_asr_aligned.set_defaults(func=run_server)



    args = parser.parse_args()
    

    # Execute the selected command's function
    if args.command:
        args.func(args)
    else:
        parser.print_help()
