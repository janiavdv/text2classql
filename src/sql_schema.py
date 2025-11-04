from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum


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


def generate_sql(query_object: Query) -> str:
    query_str = ""

    # Build SELECT clause
    if query_object.select:
        query_str += f"SELECT {', '.join(query_object.select)}"
    else:
        query_str += "SELECT *"

    # Build FROM clause
    if query_object.from_table:
        query_str += f" FROM {query_object.from_table}"

    # Build WHERE clause
    if query_object.where:
        where_conditions = []
        for predicate in query_object.where.predicates:
            where_conditions.append(
                f"{predicate.column} {predicate.operator.value} '{predicate.value}'"
            )
        if where_conditions:
            query_str += f" WHERE {' ' + query_object.where.bool_operator.value + ' '.join(where_conditions)}"

    # Build ORDER BY clause
    if query_object.order_by:
        order_by_clauses = []
        for order in query_object.order_by:
            order_by_clauses.append(
                f"{order.column} {'ASC' if order.ascending else 'DESC'}"
            )
        if order_by_clauses:
            query_str += f" ORDER BY {', '.join(order_by_clauses)}"

    return query_str
