import sys
import numpy as np
import pandas as pd

sys.path.append("..")
from config import DEFAULT_TABLE, DATA_PATH
from logs import LOGGER

# Get the vector of search
def extract_features(path, model):
    try:
        data = pd.read_csv(path)
        title_data = data['title'].tolist()
        text_data = data['text'].tolist()
        sentence_embeddings = model.sentence_encode(text_data)
        return title_data, text_data, sentence_embeddings
    except Exception as e:
        LOGGER.error(f" Error with extracting feature from question {e}")
        sys.exit(1)

# Import vectors to milvus + create local lookup table
def do_load(embedding_model, milvus_client, collection_name=DEFAULT_TABLE, data_path=DATA_PATH):
    title_data, text_data, sentence_embeddings = extract_features(data_path, embedding_model)
    ids = milvus_client.insert(collection_name, sentence_embeddings)
    milvus_client.create_index(collection_name)
    data_map = {idx: {'title':title, 'content':text} for idx, title, text in zip(ids, title_data, text_data)}
    return data_map
