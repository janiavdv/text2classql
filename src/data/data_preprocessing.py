import os
import json
from nltk.tokenize import RegexpTokenizer

DATA_PATH = "data/"
SPIDER_DATA_PATH = DATA_PATH + "spider_data/"
SPIDER_TRAIN_PATH = SPIDER_DATA_PATH + "train_spider.json"
EXCLUDE_TOKENS_PATH = DATA_PATH + "sql_exclude_tokens.txt"

DATABASE_NAMES = set()
for db_name in os.listdir(SPIDER_DATA_PATH + "database/"):
    DATABASE_NAMES.add(db_name)

with open(EXCLUDE_TOKENS_PATH, "r") as f:
    SQL_EXCLUDE_TOKENS = set(line.strip() for line in f)


def load_data(
    path: str = SPIDER_TRAIN_PATH,
    sql_exclude_tokens: set[str] = SQL_EXCLUDE_TOKENS,
    nl_tokenizer: RegexpTokenizer = RegexpTokenizer(r"\w+"),
    sql_tokenizer: RegexpTokenizer = RegexpTokenizer(r"\w+|[(),;=*<>]"),
) -> list[tuple[str, list[str], list[str]]]:
    data = []
    with open(path, "r") as f:
        for doc in json.load(f):
            database = doc["db_id"]
            question = [
                token.lower() for token in nl_tokenizer.tokenize(doc["question"])
            ]
            query = [token.lower() for token in sql_tokenizer.tokenize(doc["query"])]
            if sql_exclude_tokens:
                contains_exclude_token = any(
                    token in sql_exclude_tokens for token in query
                )
                if contains_exclude_token:
                    continue
            data.append((database, question, query))
    return data
