from data.data_preprocessing import load_data
from models.random import RandomSelectClassifier
from data.sql_dataclass import Query, convert_tokens_to_query
from data.encode_dbs import encode_nlquestion, encode_input
from tqdm import tqdm


def get_dataset(print_stats: bool = False):
    train = load_data()
    if print_stats:
        print(f"Loaded {len(train)} train data entries.")
        print(
            f"Number of unique train database IDs: {len(set([entry[0] for entry in train]))}"
        )

    dev = load_data(path="data/spider_data/dev.json")
    if print_stats:
        print(f"Loaded {len(dev)} dev data entries.")
        print(
            f"Number of unique dev database IDs: {len(set([entry[0] for entry in dev]))}"
        )

    test = load_data(path="data/spider_data/test.json")
    if print_stats:
        print(f"Loaded {len(test)} test data entries.")
        print(
            f"Number of unique test database IDs: {len(set([entry[0] for entry in test]))}"
        )

    return train, dev, test


def run_random_baseline():
    train, dev, test = get_dataset()
    model = RandomSelectClassifier()

    schemas = {}

    accuracy_sum = 0.0
    num_tests = 0
    # convert test data into the required format for prediction
    for db_id, question_tokens, query_tokens in tqdm(
        test, desc="Evaluating on test set"
    ):
        question = " ".join(question_tokens)
        encoded_question = encode_nlquestion(question)
        if db_id in schemas:
            schema = schemas[db_id]
        else:
            schema, _ = encode_input(
                f"data/spider_data/test_database/{db_id}/schema.sql", question
            )
            schemas[db_id] = schema

        if not schema:
            continue

        query: Query = convert_tokens_to_query(query_tokens)
        true_label = query.convert_to_label(schema)
        predicted_label = model.predict(encoded_question, schema)
        accuracy = sum(
            [1 for y, y_pred in zip(true_label, predicted_label) if y == y_pred]
        ) / len(true_label)
        # print(f"Accuracy: {accuracy:.2f}")
        accuracy_sum += accuracy
        num_tests += 1

    print(f"Average Accuracy: {accuracy_sum / num_tests:.2f}")


if __name__ == "__main__":
    run_random_baseline()
