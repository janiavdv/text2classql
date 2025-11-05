from data.data_preprocessing import load_data
from models.random import RandomSelectClassifier
from data.sql_dataclass import Query, convert_tokens_to_query
from data.encode_dbs import encode_nlquestion, encode_input
from tqdm import tqdm
import matplotlib.pyplot as plt


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


def run_and_plot_random_baseline():
    train, dev, test = get_dataset()
    model = RandomSelectClassifier()

    schemas = {}

    accuracies = []
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
        accuracies.append(accuracy)

    print(f"Average Accuracy: {sum(accuracies) / len(accuracies):.2f}")
    # Plotting the accuracies
    
    plt.figure(figsize=(10, 6))
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 16  # Adjust overall font size

    plt.hist(accuracies, bins=20, color='pink', edgecolor='brown')
    plt.title('Random Baseline Accuracies on Test Set', fontsize=20, fontweight='bold')
    plt.xlabel('Accuracy', fontsize=18)
    plt.ylabel('Frequency', fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    plt.tight_layout()
    plt.savefig('plots/random_baseline_accuracies.png')
    plt.show()


if __name__ == "__main__":
    run_and_plot_random_baseline()
