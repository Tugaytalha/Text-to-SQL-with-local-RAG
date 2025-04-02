import streamlit as st
import json
import pandas as pd
from src.ttsql.local import LocalContext_Ollama

@st.cache_resource(ttl=3600)
def setup_ttsql():
    if "vn" not in st.session_state:
        st.session_state.vn = LocalContext_Ollama(config={"model": "mt2sl", "path": "chroma"})
        st.session_state.vn.connect_to_sqlite("http://127.0.0.1:8001/download/flight_reservations.db")
    return st.session_state.vn

@st.cache_data(show_spinner="Generating sample questions ...")
def generate_questions_cached():
    vn = setup_ttsql()
    return vn.generate_questions()


@st.cache_data(show_spinner="Generating SQL query ...")
def generate_sql_cached(question: str):
    vn = setup_ttsql()
    return vn.generate_sql(question=question, allow_llm_to_see_data=True)

@st.cache_data(show_spinner="Getting SQL and retrieved chunks...")
def generate_sql_and_get_chunks_cached(question: str):
    """
    Get both the SQL query and the retrieved chunks used to generate it.
    Returns a tuple of (sql, chunks) where chunks is a dict containing:
    - question_sql_list: List of similar question-SQL pairs
    - ddl_list: List of related DDL statements
    - doc_list: List of related documentation
    """
    vn = setup_ttsql()
    
    # Get similar questions and their SQL
    question_sql_list = vn.get_similar_question_sql(question)
    
    # Get related DDL statements
    ddl_list = vn.get_related_ddl(question)
    
    # Get related documentation
    doc_list = vn.get_related_documentation(question)
    
    # Generate the SQL query
    sql = vn.generate_sql(question=question, allow_llm_to_see_data=True, question_sql_list=question_sql_list, ddl_list=ddl_list, doc_list=doc_list)
    
    # Package the chunks
    chunks = {
        "question_sql_list": question_sql_list,
        "ddl_list": ddl_list,
        "doc_list": doc_list
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

@st.cache_data(show_spinner="Fetching training data...")
def get_training_data_cached():
    vn = setup_ttsql()
    return vn.get_training_data()

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
def process_json_file_cached(json_data):
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
            description = row['description']
            
            # Create column definition with comment
            col_def = f"{col_name} {data_type} COMMENT '{description}'"
            column_defs.append(col_def)
        
        # Create the full CREATE TABLE statement
        ddl = f"CREATE TABLE {table_name} (\n  " + ",\n  ".join(column_defs) + "\n);"
        ddl_statements.append(ddl)
    
    return ddl_statements