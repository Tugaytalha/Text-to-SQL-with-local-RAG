import streamlit as st

from src.ttsql.local import LocalContext_Ollama

@st.cache_resource(ttl=3600)
def setup_ttsql():
    if "vn" not in st.session_state:
        st.session_state.vn = LocalContext_Ollama(config={"model": "gemma3", "path": "chroma"})
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