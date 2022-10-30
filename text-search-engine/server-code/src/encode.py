from deepsparse import Pipeline, Scheduler
from queue import Queue
from config import MODEL_PATH, ENGINE, SCHEDULE_TYPE, NUM_STREAMS, SEQUENCE_LENGTH, BATCH_SIZE
from sklearn.preprocessing import normalize
import numpy as np
import time

class SentenceModel:
    def __init__(self, 
        model_path=MODEL_PATH,
        engine=ENGINE,
        num_streams=NUM_STREAMS,
        schedule_type=SCHEDULE_TYPE,
        sequence_length=SEQUENCE_LENGTH,
        batch_size=BATCH_SIZE,
        timing=True
    ):
        self._embedding_pipeline = Pipeline.create(
            task="embedding_extraction",
            extraction_strategy="reduce_mean",
            sequence_length=sequence_length,
            batch_size=batch_size,
            model_path=model_path,
            scheduler=Scheduler.from_str(schedule_type),
            executor=num_streams,
            engine_type=engine,
        )
        
        self._timing = timing
        if self._timing:
            self._time_queue = Queue()

    def sentence_encode(self, data):
        start = time.perf_counter()
        embedding = np.array(self._embedding_pipeline(data).embeddings)
        sentence_embeddings = normalize(embedding).tolist()
        end = time.perf_counter()

        if self._timing:
            self._time_queue.put([start, end])
        
        return sentence_embeddings

    def compute_latency(self):
        batch_times = list(self._time_queue.queue)
        if len(batch_times) == 0:
            return {
                "msg" : "Latency data has been cleared"
            }

        batch_times_ms = [
            (batch_time[1] - batch_time[0]) * 1000 for batch_time in batch_times
        ]
                
        self._time_queue.queue.clear()
        
        return {
            "count" : len(batch_times),
            "median": np.median(batch_times_ms),
            "mean": np.mean(batch_times_ms),
            "std": np.std(batch_times_ms)
        }