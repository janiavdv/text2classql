import sqlparse
from transformers import AutoTokenizer


def encode_schema(
    schema_sql: str, tokenizer_name: str = "bert-base-uncased", max_length: int = 512
) -> tuple[str, dict]:
    """
    Encode a SQL schema into a tokenized, schema-agnostic embedding.

    Args:
        schema_sql (str): Full SQL schema (contains CREATE TABLE statements).
        tokenizer_name (str): HuggingFace tokenizer name.

    Returns:
        tuple[str, dict]: A tuple containing:
            - schema_text (str): Schema in textual format "table1(col1, col2); table2(col1, col2); ..."
            - encoded (dict): Tokenized representation of the schema.
    """
    statements = sqlparse.split(schema_sql)
    tables = {}

    for s in statements:
        s = s.strip()
        if not s.lower().startswith("create table"):
            continue

        parsed = sqlparse.parse(s)[0]
        print(parsed.tokens)
        tokens = [t for t in parsed.tokens if not t.is_whitespace]

        # Extract table name
        table_name = None
        for token in tokens:
            if type(token) is sqlparse.sql.Identifier:
                table_name = token.value.strip('"`')
                break
        if table_name is None:
            raise ValueError("Table name not found in CREATE TABLE statement.")

        # Extract column definitions
        cols = []
        for token in tokens:
            if type(token) is sqlparse.sql.Parenthesis:
                inside = token.value.strip("()")  # Get content inside parentheses
                for attribute in inside.split(","):
                    attribute = attribute.strip()
                    if not attribute or attribute.lower().startswith(
                        ("primary key", "foreign key")
                    ):
                        continue
                    col_name = attribute.split()[0].strip('"`')
                    cols.append(col_name)

        tables[table_name] = cols

    schema_text = (
        "; ".join([f"{table}({', '.join(cols)})" for table, cols in tables.items()])
        + ";"
    )
    print("Schema Text:", schema_text)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    encoded = tokenizer(
        schema_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )

    return schema_text, encoded


if __name__ == "__main__":
    sample_schema = """
    CREATE TABLE employees (
        id INT PRIMARY KEY,
        name VARCHAR(100),
        department_id INT,
        salary FLOAT
    );

    CREATE TABLE departments (
        id INT PRIMARY KEY,
        name VARCHAR(100)
    );
    """
    result = encode_schema(sample_schema)
    print("Schema Text:", result["schema_text"])
    print("Encoded Input IDs:", result["encoded"]["input_ids"])
    print("Encoded Attention Mask:", result["encoded"]["attention_mask"])
