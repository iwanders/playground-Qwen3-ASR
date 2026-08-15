#!/usr/bin/env python3
from aiohttp import web
import aiohttp
import torchaudio

INDEX_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Stream all the audio</title>
</head>

<body>
    <script>
    let reconnectTimer = null;
    let retryCount = 0;
    const MAX_RETRY_DELAY = 1000; // Cap the delay at 10 seconds
// index.html
async function startStreaming() {
    if (reconnectTimer) {
    clearTimeout(reconnectTimer);
    reconnectTimer = null;
    }
  // 1. Connect to the WebSocket server
  //const socket = new WebSocket('ws://localhost:8080');
  const wsProtocol = location.protocol === 'https:' ? 'wss://' : 'ws://';
  const wsUrl = wsProtocol + location.host + '/ws';
  const socket = new WebSocket(wsUrl);
  let mediaRecorder = null;
  let stream = null;

  socket.onopen = async () => {
    console.log('Connected to server, starting audio capture...');

    try {
      // 2. Request microphone access
      stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      
      // 3. Initialize MediaRecorder (webm is native for most browsers)
      mediaRecorder = new MediaRecorder(stream, {
        mimeType: 'audio/webm;codecs=opus'
      });

      // 4. Send audio chunks as soon as they are available
      mediaRecorder.ondataavailable = async (event) => {
        if (event.data.size > 0 && socket.readyState === WebSocket.OPEN) {
          // Convert Blob chunk to ArrayBuffer before sending
          const buffer = await event.data.arrayBuffer();
          socket.send(buffer);
        }
      };

      // 5. Collect and emit data every 250 milliseconds
      mediaRecorder.start(250); 
      
    } catch (err) {
      console.error('Microphone access denied or unsupported:', err);
    }
  };
  
  socket.onclose = () => {
    console.log('WebSocket closed.');

    // 1. Cleanly stop the hardware microphone if it was active
    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
      mediaRecorder.stop();
    }
    if (stream) {
      stream.getTracks().forEach(track => track.stop());
    }

    // 2. Calculate exponential backoff delay (e.g., 1s, 2s, 4s, 8s...)
    const delay = Math.min(100 * Math.pow(2, retryCount), MAX_RETRY_DELAY);
    retryCount++;

    console.log(`Attempting reconnection in ${delay / 1000} seconds...`);
    
    // 3. Start the timer to try again
    reconnectTimer = setTimeout(() => {
      startStreaming();
    }, delay);
  };

}

startStreaming();
</script>
</body>
</html>
"""

async def handle(request):
    #name = request.match_info.get('name', "Anonymous")
    #text = "Hello, " + name
    return web.Response(text=INDEX_HTML, content_type='text/html')


from torchcodec.decoders import AudioDecoder
import asyncio

import io
import threading

class UpdatableStream(io.RawIOBase):
    def __init__(self):
        self._buffer = bytearray()
        self._condition = threading.Condition()
        self._closed = False

    def readable(self):
        return True

    def writable(self):
        return True

    def readinto(self, b):
        """Reads data into the provided bytearray buffer."""
        with self._condition:
            # Wait until there is data to read or the stream is closed
            while not self._buffer and not self._closed:
                self._condition.wait()
            
            if not self._buffer and self._closed:
                return 0  # EOF reached

            # Extract data matching the requested read size
            bytes_to_read = min(len(b), len(self._buffer))
            b[:bytes_to_read] = self._buffer[:bytes_to_read]
            del self._buffer[:bytes_to_read]
            return bytes_to_read

    def write(self, b):
        """Accepts updates and appends new data to the stream."""
        with self._condition:
            if self._closed:
                raise ValueError("I/O operation on closed file.")
            self._buffer.extend(b)
            self._condition.notify_all()
            return len(b)

    def close(self):
        """Signals to the reader that no more updates will be sent."""
        with self._condition:
            self._closed = True
            self._condition.notify_all()
        super().close()

class AsyncAudioDecoderWrapper:
    def __init__(self):
        self.feeder = None
        self.decoder = None
    def setup(self):
        self.feeder = UpdatableStream()
        self.decoder = AudioDecoder(self.feeder)
        
import time 
def blocking_io_task(audio_wrapper, stop_signal):
    print( "Task doing things started (sync)...")
    audio_wrapper.setup()
    print( "Going into service")
    while not stop_signal.is_set():
        happened = stop_signal.wait(timeout=0.1)
        if happened:
            break 
        print("Task operating")
    print("Exiting task")
    return  "Result from task"



    
async def websocket_handler(request):

    ws = web.WebSocketResponse()
    await ws.prepare(request)
    audio_handler = AsyncAudioDecoderWrapper()
    stop_signal = threading.Event()
    bg_task = asyncio.create_task(asyncio.to_thread(blocking_io_task, audio_handler, stop_signal))
  

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
    # 3. Explicitly gather or await the result when you actually need it
    stop_signal.set()
    result = await bg_task
    print('websocket connection closed')

    return ws

app = web.Application()
app.add_routes([web.get('/', handle),
                web.get('/{name}', handle), web.get('/ws', websocket_handler)])

if __name__ == '__main__':
    web.run_app(app, port=8000)
