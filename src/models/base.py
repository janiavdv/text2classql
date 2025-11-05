from abc import ABC, abstractmethod


class Text2ClasSQLModel(ABC):
    """
    A model that learns to converts natural language questions into SQL queries
    using a structured intermediate representation.
    """

    def __init__(self):
        pass

    @abstractmethod
    def train(self, X, db_names, schemas, y):
        pass

    @abstractmethod
    def predict(self, x, schema):
        pass
