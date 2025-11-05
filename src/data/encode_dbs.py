import sqlparse
from transformers import AutoTokenizer


def encode_schema(
    schema_sql: str, tokenizer_name: str = "bert-base-uncased", max_length: int = 512
) -> tuple[dict[str, list[str]], dict]:
    """
    Encode a SQL schema into a tokenized, schema-agnostic embedding.

    Args:
        schema_sql (str): Full SQL schema (contains CREATE TABLE statements).
        tokenizer_name (str): HuggingFace tokenizer name.

    Returns:
        tuple[dict[str, list[str]], dict]: A tuple containing:
            - tables (dict): A dictionary mapping table names to lists of column names.
            - encoded (dict): Tokenized representation of the schema.
    """
    statements = sqlparse.split(schema_sql)
    tables = {}

    for s in statements:
        s = s.strip().lower()
        parsed = sqlparse.parse(s)[0]

        if parsed.get_type() != "CREATE" or "table" not in s:
            if s.startswith("create"):
                print(f"Warning: Skipping non-CREATE TABLE statement:\n{s}")
            continue

        tokens = [t for t in parsed.tokens if not t.is_whitespace]

        # Extract table name

        name_idx = None
        for i in range(len(tokens)):
            # if open parenthesis is found, the token before it is the table name
            if i > 0 and type(tokens[i]) is sqlparse.sql.Parenthesis:
                name_idx = i - 1
                break

        if not name_idx:
            raise ValueError("CREATE TABLE statement not properly formed.")

        if type(tokens[name_idx]) is sqlparse.sql.Identifier or (
            type(tokens[name_idx]) is sqlparse.sql.Token
        ):
            table_name = tokens[name_idx].value.lower().strip('"`')
        else:
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
                        # TODO: expand this to potentially include author(aid [PK], oid [FK→organization.oid])
                        continue
                    col_name = attribute.split()[0].strip('"`')
                    cols.append(col_name)

        tables[table_name] = cols

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    encoded = tokenizer(
        get_schema_text(tables),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )

    return tables, encoded


def get_schema_text(tables: dict[str, list[str]]) -> str:
    return (
        "; ".join([f"{table}({', '.join(cols)})" for table, cols in tables.items()])
        + ";"
    )


def encode_nlquestion(nl_question: str, tokenizer_name: str = "bert-base-uncased"):
    """Encode the natural language question into a tokenized input.

    Args:
        nl_question (str): The natural language question to be encoded.
        tokenizer_name (str, optional): HuggingFace tokenizer name.

    Returns:
        dict: Tokenized representation of the question.
    """
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    encoded = tokenizer(
        nl_question,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    )
    return encoded


def encode_input(
    schema_filename: str, nl_question: str, tokenizer_name: str = "bert-base-uncased"
) -> tuple[dict[str, list[str]], dict]:
    """Encode the database schema and the natural language question into a tokenized input.

    Args:
        schema_filename (str): Path to the file containing the SQL schema.
        nl_question (str): The natural language question to be encoded.
        tokenizer_name (str, optional): HuggingFace tokenizer name.

    Returns:
        tuple[dict[str, list[str]], dict]: A tuple containing:
            - tables (dict): A dictionary mapping table names to lists of column names.
            - encoded (dict): Tokenized representation of the combined schema and question.
    """
    try:
        with open(schema_filename, "r") as f:
            schema_sql = f.read()
    except FileNotFoundError:
        print(f"Schema file {schema_filename} not found.")
        return {}, {}

    tables, _ = encode_schema(schema_sql, tokenizer_name)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    encoded = tokenizer(
        get_schema_text(tables) + " " + nl_question,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    )
    return tables, encoded


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
    tables, encoded = encode_schema(sample_schema)
    print("Schema Text:", get_schema_text(tables))
    print("Encoded Input IDs:", encoded["input_ids"])
    print("Encoded Attention Mask:", encoded["attention_mask"])
