


import threading
from queue import Queue
import asyncio

# https://docs.python.org/3/library/asyncio-queue.html
# asyncio queues are designed to be similar to classes of the queue module. Although asyncio queues are not thread-safe, they are designed to be used specifically in async/await code.
# Ehh, so... what non blocking primitive do we have to bridge between threads and async?

# The LLM generated these two marvels :|
class ThreadToAsyncSPSC:
    def __init__(self, loop: asyncio.AbstractEventLoop):
        self.loop = loop
        self.queue = asyncio.Queue()

    # Called from background OS thread
    def produce(self, item):
        print(f"pushing {item} onto queue");
        # Safely schedules queue insertion on the event loop
        self.loop.call_soon_threadsafe(self.queue.put_nowait, item)

    # Called from asyncio coroutine
    async def consume(self):
        res =  await self.queue.get()
        print(f"consumed {res} from queue")
        return res

class AsyncToThreadSPSC:
    def __init__(self, loop: asyncio.AbstractEventLoop):
        self.loop = loop
        self.queue = asyncio.Queue()

    # Called from asyncio coroutine
    async def produce(self, item):
        await self.queue.put(item)

    # Called from background OS thread
    def consume(self):
        # Schedules queue.get() and blocks the thread until the future resolves
        future = asyncio.run_coroutine_threadsafe(self.queue.get(), self.loop)
        return future.result()
        
    def issue_task_done(self):
        self.queue.task_done()

class PipelineWorker:
    def __init__(self, loop: asyncio.AbstractEventLoop):
        self.async_to_thread = AsyncToThreadSPSC(loop)
        self.thread_to_async = ThreadToAsyncSPSC(loop)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while True:
            item = self.async_to_thread.consume()
            if item is None:
                self.async_to_thread.issue_task_done()
                break
            
            # Do work
            result = self.process(item)
            
            self.thread_to_async.produce(result)
            self.async_to_thread.issue_task_done()

    def process(self, item):
        print("doing work")
        return item * 2  # Example work

    def close(self):
        asyncio.run(self.async_to_thread.produce(None))
        self._thread.join()

    async def push(self, item):
        await self.async_to_thread.produce(item)

    async def retrieve(self):
        return await self.thread_to_async.consume()
        
if __name__ == "__main__":
    print("test")
    
    # 1. Create the loop object
    loop = asyncio.new_event_loop()
    
    # 2. Set it as the current loop for the thread
    asyncio.set_event_loop(loop)

    pipeline = PipelineWorker(loop)

        
    async def main():
        for i in range(3):
            await pipeline.push(i)
            await asyncio.sleep(1)
            res = await pipeline.retrieve()
            print(res)
        print("Manually managed loop!")
    
    
    try:
        # 3. Run your coroutine until completion
        loop.run_until_complete(main())
    finally:
        # 4. Clean up and close the loop properly
        loop.close()
