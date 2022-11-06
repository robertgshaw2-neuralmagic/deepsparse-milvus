############### Raw Data File #####################
DATA_PATH = "../data/example.csv"

############### Number of log files ###############
LOGS_NUM = 0

############### Milvus Configuration ###############
MILVUS_HOST = "127.0.0.1"
MILVUS_PORT = 19530
VECTOR_DIMENSION = 768
METRIC_TYPE = "L2"
INDEX_TYPE = "IVF_SQ8"
NLIST = 1024
DEFAULT_TABLE = "test_table"
TOP_K = 9
NPROBE = 10

############## Model Configuration #################
MODEL_URL = "http://localhost:5543/predict"