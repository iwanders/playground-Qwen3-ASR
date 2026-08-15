


import threading
from queue import Queue
import queue
from concurrent.futures import Future as ConcurrentFuture
import asyncio
from pydantic import BaseModel, ConfigDict
from enum import Enum
from typing import Any
from .pipeline import AlignedASR



class TaskType(Enum):
    ASR_CHUNK = 1
    ASR_CHUNK_SCORES = 2
    

class AsyncTask(BaseModel):
    type: TaskType
    payload: Any
    model_config = ConfigDict(extra='allow')


from abc import ABC,abstractmethod
class WorkerAbstraction(ABC):
    @abstractmethod
    def do_work(self, task: Any) -> Any:
        pass

class PipelineAbstraction(WorkerAbstraction):
    def __init__(self, pipeline: AlignedASR):
        self._pipeline: AlignedASR = pipeline
    def do_work(self, task: AsyncTask) -> Any:
        match task.type:
            case TaskType.ASR_CHUNK:
                return self._pipeline.asr_chunk(**task.payload)
            case TaskType.ASR_CHUNK_SCORES:
                return self._pipeline.asr_chunk_scores(task)

class TestAbstraction(WorkerAbstraction):
    def __init__(self):
        pass
    def do_work(self, task: int) -> int:
        print(f"do work accepting {task}");
        import time
        time.sleep(1)
        res = task * 2
        print(f"do work completed calculation of {task} result is  {res}");
        return res
         


class PipelineWorker:
    def __init__(self, work_abstraction: WorkerAbstraction):
        self._work_queue = Queue() 
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._worker: WorkerAbstraction = work_abstraction
        self._thread.start()

    def _run(self):
        while True:
            try:
                item, future = self._work_queue.get()
                try:
                    r = self._worker.do_work(item)
                    future.set_result(r)
                except Exception as e:
                    future.set_exception(e)
                finally:
                    self._work_queue.task_done()
                    
            except queue.ShutDown as e:
                break;

    def close(self):
        self._work_queue.shutdown(immediate=True)
        self._thread.join()
        
    def enqueue(self, item) -> ConcurrentFuture:
        f = ConcurrentFuture()
        self._work_queue.put((item, f))
        return f
        

        
if __name__ == "__main__":
    print("test")
    
    # 1. Create the loop object
    loop = asyncio.new_event_loop()
    
    # 2. Set it as the current loop for the thread
    asyncio.set_event_loop(loop)

    work_abstraction = TestAbstraction()
    pipeline = PipelineWorker(work_abstraction )


    def async_to_sync_bridge(v: int) -> int:
        f = pipeline.enqueue(v)
        return f.result()
    
    async def main():
        for i in range(3): 
            result = await loop.run_in_executor(None, async_to_sync_bridge, i)
            await asyncio.sleep(0.1)
            print('Calculated result:', result)
 
        print("Manually managed loop!")
    
    
    try:
        # 3. Run your coroutine until completion
        loop.run_until_complete(main())
    finally:
        # 4. Clean up and close the loop properly
        loop.close()
