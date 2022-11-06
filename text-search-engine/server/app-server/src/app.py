import uvicorn
import argparse
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
from logs import LOGGER
from milvus_helpers import MilvusHelper
from mysql_helpers import MySQLHelper
from operations.load import do_load
from operations.search import search_milvus
from operations.count import do_count
from operations.drop import do_drop
from encode import SentenceModel

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default="../data/example.csv")

def start_server(
    data_path: str = "../data/example.csv",
    host: str = "0.0.0.0",
    port: int = 5000
):
    
    MODEL = SentenceModel()
    MILVUS_CLI = MilvusHelper()
    MYSQL_CLI = MySQLHelper()

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
            status = do_drop(MILVUS_CLI, MYSQL_CLI)
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
            count = do_load(MODEL, MILVUS_CLI, MYSQL_CLI, data_path)
            LOGGER.info(f"Successfully loaded data, total count: {count}")
            return "Successfully loaded data"
        except Exception as e:
            LOGGER.error(e)
            return {'status': False, 'msg': e}, 400


    @app.get('/search')
    async def do_search_api(query_sentence: str = None):
        try:
            ids, title, text, _ = search_milvus(query_sentence, MODEL, MILVUS_CLI, MYSQL_CLI)
            res = {}
            for idx, title_i, text_i in zip(ids, title, text):
                res[idx] = {
                    'title': title_i, 
                    'text' : text_i
                }
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
    args = parser.parse_args()
    start_server(args.data_path)
