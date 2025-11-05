from .base import Text2ClasSQLModel
import random
import numpy as np


class RandomSelectClassifier(Text2ClasSQLModel):
    """
    A simple random classifier that predicts SQL queries randomly from the training set.
    """

    def __init__(self):
        super().__init__()

    def train(self, X, db_names, schemas, y):
        pass  # no training needed for random classifier

    def predict(self, x, schema):
        """Predicts SQL queries for a batch of natural language questions.

        Args:
            x (_type_): a tensor of encoded natural language questions.
            schema (dict[str, list[dict[str, list[str]]]]): the schema information for the database.
        """
        # randomly figure out which table the question is about
        n_tables = len(schema)
        table_predictions = np.zeros((n_tables))
        random_idx = random.randint(0, n_tables - 1)
        table_predictions[random_idx] = 1

        # randomly figure out which columns from the table are involved
        max_n_cols = max(len(cols) for cols in schema.values())
        column_predictions = np.zeros((max_n_cols))
        num_cols_to_select = random.randint(1, max_n_cols)
        selected_col_indices = random.sample(range(max_n_cols), num_cols_to_select)
        for i in selected_col_indices:
            column_predictions[i] = 1

        return np.concatenate([table_predictions, column_predictions])
