import uvicorn
import argparse
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
from logs import LOGGER
from milvus_helpers import MilvusHelper
from operations.load import do_load
from operations.search import search_milvus
from operations.count import do_count
from operations.drop import do_drop
from encode import SentenceModel
from config import ENGINE, SCHEDULE_TYPE, NUM_STREAMS, SEQUENCE_LENGTH

parser = argparse.ArgumentParser()
parser.add_argument("--schedule_type", type=str, default=SCHEDULE_TYPE)
parser.add_argument("--num_streams", type=int, default=NUM_STREAMS)
parser.add_argument("--engine", type=str, default=ENGINE)
parser.add_argument("--sequence_length", type=int, default=SEQUENCE_LENGTH)

# map from MilvusIDs to Text, Title Pairs
# this should be a database endpoint in a real application
data_map = {}

def start_server(
    model_config: dict,
    host: str = "0.0.0.0",
    port: int = 5000
):
    
    MODEL = SentenceModel(**model_config)
    MILVUS_CLI = MilvusHelper()

    app = FastAPI()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"])
    
    @app.post('/latency')
    def compute_latency():
        # Compute Latency of Recent Queries + Reset Data
        try:
            stats = MODEL.compute_latency()
            LOGGER.info("Successfully computed recent query latency!")
            return stats
        except Exception as e:
            LOGGER.error(e)
            return {'status': False, 'msg': e}, 400

    @app.post('/count')
    async def count_text(table_name: str = None):
        # Returns the total number of titles in the system
        try:
            num = do_count(table_name, MILVUS_CLI)
            LOGGER.info("Successfully count the number of titles!")
            return num
        except Exception as e:
            LOGGER.error(e)
            return {'status': False, 'msg': e}, 400

    @app.post('/drop')
    async def drop_tables():
        # Delete the collection of Milvus and MySQL
        try:
            status = do_drop(MILVUS_CLI)
            data_map = {}
            LOGGER.info("Successfully drop tables in Milvus!")
            return status
        except Exception as e:
            LOGGER.error(e)
            return {'status': False, 'msg': e}, 400

    @app.post('/load')
    def load_text():
        # Insert all the image under the file path to Milvus
        try:
            data = do_load(MODEL, MILVUS_CLI)
            for idx in data:
                data_map[idx] = data[idx]
            LOGGER.info(f"Successfully loaded data, total count: {len(data_map)}")
            return "Successfully loaded data"
        except Exception as e:
            LOGGER.error(e)
            return {'status': False, 'msg': e}, 400


    @app.get('/search')
    async def do_search_api(query_sentence: str = None):
        try:
            ids, _ = search_milvus(query_sentence, MODEL, MILVUS_CLI)
            res = {}
            for idx in ids:
                res[idx] = data_map[idx]
            LOGGER.info("Successfully searched similar text!")
            return res
        except Exception as e:
            LOGGER.error(e)
            return {'status': False, 'msg': e}, 400

    # run with 1 worker process to avoid copying model
    # note: FastAPI handles concurrent request via a ThreadPool
    # note: DeepSparse Pipelines handle concurrent inferences via a ThreadPool
    #       and DeepSparse engine can handle multiple input streams
    uvicorn.run(app=app, host=host, port=port, workers=1)

if __name__ == "__main__":
    args = vars(parser.parse_args())
    start_server(model_config=args)
