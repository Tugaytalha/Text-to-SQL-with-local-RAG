import json
import pandas as pd
import numpy as np
import streamlit as st

from src.ttsql.local import LocalContext_Ollama
from src.ttsql.utils import visualize_query_embeddings, score_passed

@st.cache_resource(ttl=3600)
def setup_ttsql():
    if "vn" not in st.session_state:
        st.session_state.vn = LocalContext_Ollama(config={"model": "mannix/defog-llama3-sqlcoder-8b", "path": "chroma"})
        st.session_state.vn.connect_to_sqlite("http://127.0.0.1:8001/download/flight_reservations.db")
    return st.session_state.vn


@st.cache_data(show_spinner="Generating sample questions ...")
def generate_questions_cached():
    vn = setup_ttsql()
    return vn.generate_questions()


@st.cache_data(show_spinner="Generating SQL query ...")
def generate_sql_cached(question: str):
    vn = setup_ttsql()
    return vn.generate_sql(question=question, allow_llm_to_see_data=True, suggest_columns=True)


@st.cache_data(show_spinner="Getting SQL and retrieved chunks...")
def generate_sql_and_get_chunks_cached(question: str, rerank: bool = False):
    """
    Get the SQL query and the retrieved chunks used to generate it.
    Returns a tuple of (sql, chunks) where chunks is a dict containing:
    - question_sql_list: List of similar question-SQL pairs
    - ddl_list: List of related DDL statements
    - doc_list: List of related documentation
    - retrieved_embeddings: A numpy array of embeddings for the retrieved chunks
    """
    vn = setup_ttsql()

    # Get similar questions and their SQL
    question_sql_list = vn.get_similar_question_sql(question)

    suggest_columns = True
    # Get related DDL statements
    if suggest_columns:
        pred_cols = LocalContext_Ollama(config={"model": "mt2sql", "path": "chroma"}).suggest_columns_for_query(question)
        ddl_dic = dict() #set()
        for i, col in enumerate(pred_cols):
            # ddl_list.update()
            results = vn.get_related_ddl_with_score(col, rerank=rerank, n_results=5)
            print(results)
            for j, (chunk, score) in enumerate(results):
                if chunk not in ddl_dic.keys() and score_passed(score, rerank):
                    ddl_dic[chunk] = (str(i) + "." + str(j) + ".")
        ddl_list = list()
        for chunk, num in ddl_dic.items():
            ddl_list.append(num + chunk)

    else:
        ddl_list = vn.get_related_ddl(question)

    # Get related documentation
    doc_list = vn.get_related_documentation(question)

    # --- Generate embeddings for retrieved chunks ---
    retrieved_texts = []
    if question_sql_list:
        # Embed the question part of the pair as it's closer to the user query
        retrieved_texts.extend([qs['question'] for qs in question_sql_list])
        # Alternatively, embed the combined JSON string if that's what's stored
        # retrieved_texts.extend([json.dumps(qs) for qs in question_sql_list])
    if ddl_list:
        retrieved_texts.extend(ddl_list)
    if doc_list:
        retrieved_texts.extend(doc_list)

    retrieved_embeddings = []
    if retrieved_texts:
        # Use the embedding function directly from the vn instance
        retrieved_embeddings = vn.embedding_function(retrieved_texts)

    # ----------------------------------------------

    # Generate the SQL query using the retrieved context
    sql = vn.generate_sql(question=question, allow_llm_to_see_data=True, suggest_columns=True,
                          question_sql_list=question_sql_list,
                          ddl_list=ddl_list, doc_list=doc_list)

    # Package the chunks and their embeddings
    chunks = {
        "question_sql_list": question_sql_list,
        "ddl_list": ddl_list,
        "doc_list": doc_list,
        "retrieved_embeddings": np.array(retrieved_embeddings) # Store as numpy array
    }

    return sql, chunks


@st.cache_data(show_spinner="Checking for valid SQL ...")
def is_sql_valid_cached(sql: str):
    vn = setup_ttsql()
    return vn.is_sql_valid(sql=sql)


@st.cache_data(show_spinner="Running SQL query ...")
def run_sql_cached(sql: str):
    vn = setup_ttsql()
    return vn.run_sql(sql=sql)


@st.cache_data(show_spinner="Checking if we should generate a chart ...")
def should_generate_chart_cached(question, sql, df):
    vn = setup_ttsql()
    return vn.should_generate_chart(df=df)


@st.cache_data(show_spinner="Generating Plotly code ...")
def generate_plotly_code_cached(question, sql, df):
    vn = setup_ttsql()
    code = vn.generate_plotly_code(question=question, sql=sql, df=df)
    return code


@st.cache_data(show_spinner="Getting retrieved chunks...")
def get_retrieved_chunks_cached(question: str):
    """
    Get the chunks of information retrieved from the vector database for a given question.
    This includes DDL statements, documentation, and question-SQL pairs that are relevant to the query.
    """
    vn = setup_ttsql()
    return vn.get_retrieved_chunks(question=question)


@st.cache_data(show_spinner="Running Plotly code ...")
def generate_plot_cached(code, df):
    vn = setup_ttsql()
    return vn.get_plotly_figure(plotly_code=code, df=df)


@st.cache_data(show_spinner="Generating followup questions ...")
def generate_followup_cached(question, sql, df):
    vn = setup_ttsql()
    return vn.generate_followup_questions(question=question, sql=sql, df=df)


@st.cache_data(show_spinner="Generating summary ...")
def generate_summary_cached(question, df):
    vn = setup_ttsql()
    return vn.generate_summary(question=question, df=df)


# Database Management Functions


def get_training_data():
    vn = setup_ttsql()
    return vn.get_training_data()


@st.cache_data(show_spinner="Loading all embeddings...")
def get_all_embeddings_cached():
    """
    Retrieves all embeddings from the vector store.
    """
    vn = setup_ttsql()
    return vn.get_all_embeddings()


# No cache needed here as it depends on runtime data (query) and creates a file
def generate_visualization(query: str, chunks: dict):
    """
    Generates and saves the embedding visualization plot.

    Args:
        query (str): The user's query.
        chunks (dict): The dictionary containing retrieved chunks and their embeddings.

    Returns:
        str: Path to the saved visualization image, or None.
    """
    vn = setup_ttsql()
    try:
        # 1. Get Query Embedding
        # Use the embedding function directly for consistency
        query_embedding = vn.embedding_function([query])[0]
        query_embedding = np.array(query_embedding)

        # 2. Get All Chunk Embeddings
        all_chunk_embeddings = get_all_embeddings_cached()
        if all_chunk_embeddings.size == 0:
             st.warning("No embeddings found in the database to visualize.")
             return None

        # 3. Get Retrieved Embeddings (already calculated)
        retrieved_embeddings = chunks.get("retrieved_embeddings", np.array([]))

        # 4. Generate Visualization
        visualization_path = visualize_query_embeddings(
            query=query,
            query_embedding=query_embedding,
            all_chunk_embeddings=all_chunk_embeddings,
            retrieved_embeddings=retrieved_embeddings
        )
        return visualization_path

    except Exception as e:
        st.error(f"Failed to generate visualization: {e}")
        return None


@st.cache_data(show_spinner="Adding question-SQL pair...", ttl=1)
def add_question_sql_cached(question, sql):
    vn = setup_ttsql()
    return vn.add_question_sql(question=question, sql=sql)


@st.cache_data(show_spinner="Adding DDL statement...", ttl=1)
def add_ddl_cached(ddl):
    vn = setup_ttsql()
    return vn.add_ddl(ddl=ddl)


@st.cache_data(show_spinner="Adding documentation...", ttl=1)
def add_documentation_cached(documentation):
    vn = setup_ttsql()
    return vn.add_documentation(documentation=documentation)


@st.cache_data(show_spinner="Removing training data...", ttl=1)
def remove_training_data_cached(id):
    vn = setup_ttsql()
    return vn.remove_training_data(id=id)


@st.cache_data(show_spinner="Processing JSON file...", ttl=1)
def process_json_file_cached_by_table(json_data):
    """
    Process JSON data and convert it to DDL statements.
    Each table's columns are grouped together to create a single CREATE TABLE statement.
    """
    # Convert JSON to DataFrame
    df = pd.DataFrame(json_data)

    # Group by table_name to create DDL statements
    ddl_statements = []
    for table_name, group in df.groupby('table_name'):
        # Create column definitions
        column_defs = []
        for _, row in group.iterrows():
            col_name = row['column_name']
            data_type = row['data_type'].strip('[]')  # Remove brackets from data type
            description = "Description:" + row['description'] if row['description'] != 'nan' else ''
            usage = "Usage:" + row['usage'] if row['usage'] != 'nan' else ''
            connective = ','

            # Create column definition with comment
            col_def = f"{col_name} {data_type} COMMENT '{description}{connective}{usage}'"
            column_defs.append(col_def)

        # Create the full CREATE TABLE statement
        ddl = f"CREATE TABLE {table_name} (\n  " + ",\n  ".join(column_defs) + "\n);"
        ddl_statements.append(ddl)

    return ddl_statements


@st.cache_data(show_spinner="Processing JSON file...", ttl=1)
def process_json_file_cached(json_data):
    """
    Process JSON data and convert each record to a DDL statement.
    Each JSON entry is converted to an individual ALTER TABLE statement
    that adds a column with a comment.
    """
    # Convert JSON to DataFrame
    df = pd.DataFrame(json_data)

    ddl_statements = []
    for _, row in df.iterrows():
        table_name = row['table_name']
        column_name = row['column_name']
        data_type = row['data_type'].strip('[]')  # Remove brackets from data type
        description = "Description:" + row['description'] if row['description'] != 'nan' else ''
        usage = "Usage:" + row['usage'] if row['usage'] != 'nan' else ''
        connective = ','

        # Create an ALTER TABLE statement for each column
        ddl = f"ALTER TABLE {table_name} ADD COLUMN {column_name} {data_type} COMMENT '{description}{connective}{usage}';"
        ddl_statements.append(ddl)

    return ddl_statements


@st.cache_data(show_spinner="Removing collection...", ttl=1)
def remove_collection_cached(collection_name):
    vn = setup_ttsql()
    vn.remove_collection(collection_name=collection_name)
