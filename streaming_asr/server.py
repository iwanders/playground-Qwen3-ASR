#!/usr/bin/env python3
from aiohttp import web
import aiohttp
import torchaudio
from pathlib import Path

from torchcodec.decoders import AudioDecoder
import asyncio

import io
from qwen3_asr_support.pipeline_worker import TestAbstraction, PipelineWorker

THIS_PATH = Path(__file__).parent.absolute()

work_abstraction = TestAbstraction()
pipeline = PipelineWorker(work_abstraction )
def pipeline_entry(data):
    v = len(data)
    f = pipeline.enqueue(v)
    return f.result()




async def handle(request):
    return web.FileResponse(THIS_PATH/'index.html')

    
async def websocket_handler(request):

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
                
            result = await loop.run_in_executor(None, pipeline_entry, msg.data)
            print(result)
            # This is a float32 array.
            # Cool, convert this into ehm, the actual samples, and dispatch to the pipeline?
             
    print('websocket connection closed')

    return ws

app = web.Application()
app.add_routes([web.get('/', handle),
                web.get('/{name}', handle), web.get('/ws', websocket_handler)])

if __name__ == '__main__':
    web.run_app(app, port=8000)
