import threading, queue, time, numpy, argparse
from deepsparse import Pipeline, Scheduler
from pprint import pprint

ENGINE = "deepsparse"
MODEL_TYPE = "sparse"
SCHEDULE = "sync"
NUM_STREAMS = 0
NUM_CLIENTS = 1
BATCH_SIZE = 1
SECONDS_TO_RUN = 5.0
SEQUENCE_LENGTH = 128

model_paths = {
    'sparse'    : 'zoo:nlp/masked_language_modeling/bert-base/pytorch/huggingface/wikipedia_bookcorpus/12layer_pruned80_quant-none-vnni',
    'dense'     : 'zoo:nlp/masked_language_modeling/bert-base/pytorch/huggingface/wikipedia_bookcorpus/base-none',
}

parser = argparse.ArgumentParser()
parser.add_argument("--schedule_type", type=str, default=SCHEDULE)
parser.add_argument("--num_clients", type=int, default=NUM_CLIENTS)
parser.add_argument("--num_streams", type=int, default=NUM_STREAMS)
parser.add_argument("--seconds_to_run", type=float, default=SECONDS_TO_RUN)
parser.add_argument("--engine", type=str, default=ENGINE)
parser.add_argument("--model_type", type=str, default=MODEL_TYPE)
parser.add_argument("--sequence_length", type=int, default=SEQUENCE_LENGTH)

class ExecutorThread(threading.Thread):
    def __init__(self,
        pipeline: Pipeline,
        input: str, 
        time_queue: queue.Queue,
        max_time: float
    ):
        super(ExecutorThread, self).__init__()
        self._pipeline = pipeline
        self._input = input
        self._time_queue = time_queue
        self._max_time = max_time

    def iteration(self, input:str):
        start = time.perf_counter()
        output = self._pipeline(input)
        end = time.perf_counter()
        return output, start, end

    def run(self):
        while time.perf_counter() < self._max_time:
            _, start, end = self.iteration(self._input)
            self._time_queue.put([start, end])

def run_benchmarking(
    schedule_type: str = SCHEDULE,
    num_clients: int = NUM_CLIENTS,
    num_streams: int = NUM_STREAMS,
    seconds_to_run: float = SECONDS_TO_RUN,
    engine: str = ENGINE,
    model_type: str = MODEL_TYPE,
    sequence_length: int = SEQUENCE_LENGTH,
    input="The quick brown fox jumps over the lazy dog The quick brown fox jumps over the lazy dog The quick brown fox jumps over the lazy dog The quick brown fox jumps over the lazy dog The quick brown fox jumps over the lazy dog The quick brown fox jumps over the lazy dog The quick brown fox jumps over the lazy dog The quick brown fox jumps over the lazy dog The quick brown fox jumps over the lazy dog The quick brown fox jumps over the lazy dog The quick brown fox jumps over the lazy dog The quick brown fox jumps over the lazy dog The quick brown fox jumps over the lazy dog The quick brown fox jumps over the lazy dog The quick brown fox jumps over the lazy dog The quick brown fox jumps over the lazy dog The quick brown fox jumps over the lazy dog The quick brown fox jumps over the lazy dog The quick brown fox jumps over the lazy dog The quick brown fox jumps over the lazy dog The quick brown fox jumps over the lazy dog The quick brown fox jumps over the lazy dog The quick brown fox jumps over the lazy dog The quick brown fox jumps over the lazy dog The quick brown fox jumps over the lazy dog The quick brown fox jumps over the lazy dog The quick brown fox jumps over the lazy dog The quick brown fox jumps over the lazy dog The quick brown fox jumps over the lazy dog The quick brown fox jumps over the lazy dog The quick brown fox jumps over the lazy dog The quick brown fox jumps over the lazy dog The quick brown fox jumps over the lazy dog The quick brown fox jumps over the lazy dog The quick brown fox jumps over the lazy dog The quick brown fox jumps over the lazy dog The quick brown fox jumps over the lazy dog The quick brown fox jumps over the lazy dog The quick brown fox jumps over the lazy dog The quick brown fox jumps over the lazy dog The quick brown fox jumps over the lazy dog The quick brown fox jumps over the lazy dog The quick brown fox jumps over the lazy dog The quick brown fox jumps over the lazy dog The quick brown fox jumps over the lazy dog The quick brown fox jumps over the lazy dog The quick brown fox jumps over the lazy dog The quick brown fox jumps over the lazy dog The quick brown fox jumps over the lazy dog The quick brown fox jumps over the lazy dog The quick brown fox jumps over the lazy dog The quick brown fox jumps over the lazy dog The quick brown fox jumps over the lazy dog "
):
    
    embedding_pipeline = Pipeline.create(
        task="embedding_extraction",
        model_path=model_paths[model_type],
        extraction_strategy="reduce_mean",
        scheduler=Scheduler.from_str(schedule_type),
        executor=num_streams,
        engine_type=engine,
        batch_size=BATCH_SIZE,
        sequence_length=sequence_length
    )
    
    time_queue = queue.Queue()
    max_time = time.perf_counter() + seconds_to_run
    
    threads = []
    for _ in range(num_clients):
        threads.append(ExecutorThread(embedding_pipeline, input, time_queue, max_time))

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()
    
    batch_times = list(time_queue.queue)
    batch_times_ms = [
        (batch_time[1] - batch_time[0]) * 1000 for batch_time in batch_times
    ]

    # Calculate statistics
    # Note: We want to know all of the executions that could be performed within a
    # given amount of wallclock time. This calculation as-is includes the test overhead
    # such as saving timing results for each iteration so it isn't a best-case but is a
    # realistic case.
    first_start_time = min([b[0] for b in batch_times])
    last_end_time = max([b[1] for b in batch_times])
    total_time_executing = last_end_time - first_start_time
    items_per_sec = (BATCH_SIZE * len(batch_times)) / total_time_executing

    benchmark_dict = {
        "scenario": schedule_type,
        "num_clients": num_clients,
        "num_streams": num_streams,
        "items_per_sec": items_per_sec,
        "seconds_ran": total_time_executing,
        "iterations": len(batch_times_ms),
        "median": numpy.median(batch_times_ms),
        "mean": numpy.mean(batch_times_ms),
        "std": numpy.std(batch_times_ms)
    }
    pprint(benchmark_dict)

if __name__ == "__main__":
    args = vars(parser.parse_args())
    print("Running with:")
    pprint(args)
    run_benchmarking(**args)