from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum
import numpy as np


class Operator(Enum):
    EQUAL = "="
    NOT_EQUAL = "!="
    GREATER_THAN = ">"
    LESS_THAN = "<"
    GREATER_EQUAL = ">="
    LESS_EQUAL = "<="


class BoolOperator(Enum):
    AND = "AND"
    OR = "OR"


@dataclass(frozen=True)
class Predicate:
    column: str  # column name involved in the predicate
    operator: Operator  # operator used in the predicate (e.g., '=', '>', '<', etc.)
    value: str | int | float  # value to compare the column against


@dataclass(frozen=True)
class Where:
    predicates: List[Predicate] = field(
        default_factory=list
    )  # list of predicates for filtering
    bool_operator: BoolOperator = (
        BoolOperator.AND
    )  # logical operator to combine predicates ("AND" or "OR")


# TODO: support nested conditions


@dataclass(frozen=True)
class OrderBy:
    column: str  # column name to order by
    ascending: bool = True  # True for ascending order, False for descending


@dataclass(frozen=True)
class Query:
    select: List[str] = field(default_factory=list)  # list of columns for projection
    from_table: str = ""  # table to query from
    where: Where = field(default_factory=Where)  # filtering conditions
    order_by: Optional[List[OrderBy]] = field(
        default_factory=list
    )  # list of columns to order the results by
    limit: Optional[int] = None  # limit on number of results

    def convert_to_label(self, schema) -> np.ndarray:
        """Converts a Query object into a label representation using the provided database schema.

        Args:
            query_object (self): The query object to convert.
            schema (dict[str, list[str]]): The database schema. Table name -> list of column names.

        Returns:
            np.ndarray: A label representation of the query.
        """
        # error checking
        if not self.from_table:
            raise ValueError("FROM table is not specified in the query.")

        table_labels = np.zeros((len(schema)))
        column_labels = np.zeros((max([len(cols) for cols in schema.values()])))

        # Table selection
        if self.from_table in schema:
            table_idx = list(schema.keys()).index(self.from_table)
            table_labels[table_idx] = 1
        else:
            print(schema)
            raise ValueError(f"Table {self.from_table} not found in schema.")
        # Column selection
        table_columns = schema[self.from_table]
        if not self.select:
            # select all columns
            for i in range(len(table_columns)):
                column_labels[i] = 1
        else:
            for col in self.select:
                if col in table_columns:
                    col_idx = table_columns.index(col)
                    column_labels[col_idx] = 1

        return np.concatenate([table_labels, column_labels])

    def generate_sql(self) -> str:
        query_str = ""

        # Build SELECT clause
        if self.select:
            query_str += f"SELECT {', '.join(self.select)}"
        else:
            query_str += "SELECT *"

        # Build FROM clause
        if self.from_table:
            query_str += f" FROM {self.from_table}"
        else:
            raise ValueError("FROM table is not specified in the query.")

        # Build WHERE clause
        if self.where:
            where_conditions = []
            for predicate in self.where.predicates:
                where_conditions.append(
                    f"{predicate.column} {predicate.operator.value} '{predicate.value}'"
                )
            if where_conditions:
                query_str += f" WHERE {' ' + self.where.bool_operator.value + ' '.join(where_conditions)}"

        # Build ORDER BY clause
        if self.order_by:
            order_by_clauses = []
            for order in self.order_by:
                order_by_clauses.append(
                    f"{order.column} {'ASC' if order.ascending else 'DESC'}"
                )
            if order_by_clauses:
                query_str += f" ORDER BY {', '.join(order_by_clauses)}"

        # Build LIMIT clause
        if self.limit:
            query_str += f" LIMIT {self.limit}"

        return query_str + ";"


def convert_tokens_to_query(tokens: list[str]) -> Query:
    """Converts a list of tokens representing a SQL query into a Query dataclass object.

    Args:
        tokens (list[str]): List of tokens representing the SQL query.

    Returns:
        Query: The corresponding Query dataclass object.
    """
    first_non_select_idx = len(tokens)
    where = tokens.index("where") if "where" in tokens else float("inf")
    order = tokens.index("order") if "order" in tokens else float("inf")
    limit = tokens.index("limit") if "limit" in tokens else float("inf")
    first_non_select_idx = min(where, order, limit, first_non_select_idx)

    columns, from_table = convert_select_tokens(tokens[:first_non_select_idx])

    ## TODO: parse WHERE, ORDER BY, LIMIT clauses

    query = Query(select=columns, from_table=from_table)
    return query


def convert_select_tokens(select_tokens: list[str]) -> tuple[List[str], str]:
    select_columns, from_table = [], None
    cursor = 0
    while cursor < len(select_tokens):
        token = select_tokens[cursor]
        if token == "from":
            cursor += 1
            from_table = select_tokens[cursor]
            break
        if token != "select" and token != ",":
            select_columns.append(token)
        cursor += 1
    return select_columns, from_table
