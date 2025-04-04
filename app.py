import time
import streamlit as st
import json
from ttsql_calls import (
    generate_questions_cached,
    # generate_sql_cached,
    run_sql_cached,
    generate_plotly_code_cached,
    generate_plot_cached,
    generate_followup_cached,
    should_generate_chart_cached,
    is_sql_valid_cached,
    generate_summary_cached,
    setup_ttsql,
    get_training_data,
    add_question_sql_cached,
    add_ddl_cached,
    add_documentation_cached,
    remove_training_data_cached,
    process_json_file_cached,
    # get_retrieved_chunks_cached,
    generate_sql_and_get_chunks_cached,
    remove_collection_cached,
    generate_visualization,
)

from .src.ttsql.utils import (
    visualize_query_embeddings,
)

avatar_url = "https://play-lh.googleusercontent.com/27WE_FCTH2aJh0mzYmPYgQp6ZdmZK27Vyf2ER_o9862cAE2L_tWikyx9qsMntI3Nbw"

st.set_page_config(layout="wide")
st.title("Albaraka Text2SQL")

# Create tabs for the app
tab1, tab2 = st.tabs(["Query Interface", "Database Management"])

with tab1:
    st.sidebar.title("Output Settings")
    st.sidebar.checkbox("Show Retrieved Chunks", value=True, key="show_chunks")
    st.sidebar.checkbox("Show SQL", value=True, key="show_sql")
    st.sidebar.checkbox("Show Embedding Visualization", value=True, key="show_viz")
    st.sidebar.checkbox("Show Table", value=False, key="show_table")
    st.sidebar.checkbox("Show Plotly Code", value=False, key="show_plotly_code")
    st.sidebar.checkbox("Show Chart", value=False, key="show_chart")
    st.sidebar.checkbox("Show Summary", value=False, key="show_summary")
    st.sidebar.checkbox("Show Follow-up Questions", value=False, key="show_followup")
    st.sidebar.button("Reset", on_click=lambda: set_question(None), use_container_width=True)


    def set_question(question):
        st.session_state["my_question"] = question


    assistant_message_suggested = st.chat_message(
        "assistant", avatar=avatar_url
    )
    if assistant_message_suggested.button("Click to show suggested questions"):
        st.session_state["my_question"] = None
        questions = generate_questions_cached()
        for i, question in enumerate(questions):
            time.sleep(0.05)
            button = st.button(
                question,
                on_click=set_question,
                args=(question,),
            )

    my_question = st.session_state.get("my_question", default=None)

    if my_question is None:
        my_question = st.chat_input(
            "Ask me a question about your data",
        )

    if my_question:
        st.session_state["my_question"] = my_question
        user_message = st.chat_message("user")
        user_message.write(f"{my_question}")

        # Generate SQL
        sql, chunks = generate_sql_and_get_chunks_cached(question=my_question)

        # Show the Retrieved chunks
        if st.session_state.get("show_chunks", False):
            # Check if chunks dictionary is not None and has content
            if chunks and (chunks.get("question_sql_list") or chunks.get("ddl_list") or chunks.get("doc_list")):
                chunks_message = st.chat_message("assistant", avatar=avatar_url)
                chunks_message.subheader("Retrieved Context")

                # Display similar questions and their SQL
                if chunks.get("question_sql_list"):
                    with chunks_message.expander("Similar Questions and SQL"):
                        for i, qs in enumerate(chunks["question_sql_list"], 1):
                            st.markdown(f"**Similar Question {i}:** {qs['question']}")
                            st.markdown(f"**SQL:**\n```sql\n{qs['sql']}\n```")

                # Display related DDL statements
                if chunks.get("ddl_list"):
                    with chunks_message.expander("Related DDL Statements"):
                        for i, ddl in enumerate(chunks["ddl_list"], 1):
                            st.markdown(f"**DDL {i}:**\n```sql\n{ddl}\n```")

                # Display related documentation
                if chunks.get("doc_list"):
                    with chunks_message.expander("Related Documentation"):
                        for i, doc in enumerate(chunks["doc_list"], 1):
                            st.markdown(f"**Documentation {i}:**\n{doc}")
            elif st.session_state.get("show_chunks"):  # Only show message if checkbox is ticked
                chunks_message = st.chat_message("assistant", avatar=avatar_url)
                chunks_message.info("No relevant context was retrieved from the database for this query.")

        if sql:
            if is_sql_valid_cached(sql=sql):
                # Only show SQL if the option is enabled
                if st.session_state.get("show_sql", True):
                    assistant_message_sql = st.chat_message(
                        "assistant", avatar=avatar_url
                    )
                    assistant_message_sql.code(sql, language="sql", line_numbers=True)
            else:
                # Always show error messages
                assistant_message = st.chat_message(
                    "assistant", avatar=avatar_url
                )
                assistant_message.write(sql)
                st.stop()

            # Embedding visualization as a graph
            if st.session_state.get("show_viz", False):
                viz_message = st.chat_message("assistant", avatar=avatar_url)
                with viz_message:  # Use 'with' to keep spinner within the message
                    with st.spinner("Generating embedding visualization..."):
                        visualization_path = generate_visualization(my_question, chunks)
                        if visualization_path:
                            st.image(visualization_path, caption="Query and Chunk Embeddings (UMAP Projection)")
                        else:
                            # Display message only if visualization failed or was skipped
                            if chunks and chunks.get(
                                    "retrieved_embeddings") is not None:  # Check if chunks were actually processed
                                st.warning(
                                    "Could not generate embedding visualization (check console/logs for details). This might happen if there are too few data points in the database.")

            # Run SQL (this is always needed for subsequent operations)
            df = run_sql_cached(sql=sql)

            if df is not None:
                st.session_state["df"] = df

                # Only show table if the option is enabled
                if st.session_state.get("show_table", True):
                    df_to_display = st.session_state.get("df")
                    assistant_message_table = st.chat_message(
                        "assistant",
                        avatar=avatar_url,
                    )
                    if len(df_to_display) > 10:
                        assistant_message_table.text("First 10 rows of data")
                        assistant_message_table.dataframe(df_to_display.head(10))
                    else:
                        assistant_message_table.dataframe(df_to_display)

                # Only check if we should generate a chart if we need to show either the chart or the code
                chart_should_be_generated = False
                if st.session_state.get("show_chart", True) or st.session_state.get("show_plotly_code", False):
                    chart_should_be_generated = should_generate_chart_cached(question=my_question, sql=sql, df=df)

                code = None
                if chart_should_be_generated:
                    # Only generate plotly code if we need to show either the chart or the code
                    code = generate_plotly_code_cached(question=my_question, sql=sql, df=df)

                    # Only show plotly code if the option is enabled
                    if st.session_state.get("show_plotly_code", False) and code is not None and code != "":
                        assistant_message_plotly_code = st.chat_message(
                            "assistant",
                            avatar=avatar_url,
                        )
                        assistant_message_plotly_code.code(
                            code, language="python", line_numbers=True
                        )

                    # Only generate and show chart if the option is enabled
                    if st.session_state.get("show_chart", True) and code is not None and code != "":
                        assistant_message_chart = st.chat_message(
                            "assistant",
                            avatar=avatar_url,
                        )
                        fig = generate_plot_cached(code=code, df=df)
                        if fig is not None:
                            assistant_message_chart.plotly_chart(fig)
                        else:
                            assistant_message_chart.error("I couldn't generate a chart")

                # Only generate and show summary if the option is enabled
                if st.session_state.get("show_summary", True):
                    assistant_message_summary = st.chat_message(
                        "assistant",
                        avatar=avatar_url,
                    )
                    summary = generate_summary_cached(question=my_question, df=df)
                    if summary is not None:
                        assistant_message_summary.text(summary)

                # Only generate and show followup questions if the option is enabled
                if st.session_state.get("show_followup", True):
                    assistant_message_followup = st.chat_message(
                        "assistant",
                        avatar=avatar_url,
                    )
                    followup_questions = generate_followup_cached(
                        question=my_question, sql=sql, df=df
                    )
                    st.session_state["df"] = None

                    if len(followup_questions) > 0:
                        assistant_message_followup.text(
                            "Here are some possible follow-up questions"
                        )
                        # Print the first 5 follow-up questions
                        for question in followup_questions[:5]:
                            assistant_message_followup.button(question, on_click=set_question, args=(question,))

        else:
            assistant_message_error = st.chat_message(
                "assistant", avatar=avatar_url
            )
            assistant_message_error.error("I wasn't able to generate SQL for that question")

# Database Management Tab
with tab2:
    st.title("Database Management")

    # Initialize the Text-to-SQL model
    vn = setup_ttsql()

    # Create three columns for the training data types
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.subheader("Add Question-SQL Pair")
        question_input = st.text_area("Question", height=100, key="question_sql_pair")
        sql_input = st.text_area("SQL Query", height=150, key="sql_input")
        if st.button("Add Question-SQL Pair"):
            if question_input and sql_input:
                id = add_question_sql_cached(question=question_input, sql=sql_input)
                st.success(f"Added Question-SQL pair with ID: {id}")
                # Clear inputs after successful addition
                # st.session_state["question_sql_pair"] = ""
                # st.session_state["sql_input"] = ""
                # Refresh training data
                st.session_state["training_data"] = get_training_data()
            else:
                st.error("Both Question and SQL Query are required.")

    with col2:
        st.subheader("Add DDL")
        ddl_input = st.text_area("DDL Statement", height=250, key="ddl_input")
        if st.button("Add DDL"):
            if ddl_input:
                id = add_ddl_cached(ddl=ddl_input)
                st.success(f"Added DDL with ID: {id}")
                # Clear input after successful addition
                # st.session_state["ddl_input"] = ""
                # Refresh training data
                st.session_state["training_data"] = get_training_data()
            else:
                st.error("DDL Statement is required.")

    with col3:
        st.subheader("Add Documentation")
        doc_input = st.text_area("Documentation", height=250, key="doc_input")
        if st.button("Add Documentation"):
            if doc_input:
                id = add_documentation_cached(documentation=doc_input)
                st.success(f"Added Documentation with ID: {id}")
                # Clear input after successful addition
                # st.session_state["doc_input"] = ""
                # Refresh training data
                st.session_state["training_data"] = get_training_data()
            else:
                st.error("Documentation is required.")

    # Add JSON Upload Section
    with col4:
        st.subheader("Upload JSON Schema")
        uploaded_file = st.file_uploader("Choose a JSON file", type=['json'])

    if uploaded_file is not None:
        try:
            # Read and parse JSON file
            json_data = json.load(uploaded_file)

            # Process the JSON data and convert to DDL statements
            ddl_statements = process_json_file_cached(json_data)

            # Display the generated DDL statements
            st.subheader("Generated DDL Statements")
            for i, ddl in enumerate(ddl_statements):
                st.code(ddl, language="sql", line_numbers=True)

            # Add a button to add all DDL statements
            if st.button("Add All DDL Statements"):
                success_count = 0
                for ddl in ddl_statements:
                    try:
                        id = add_ddl_cached(ddl=ddl)
                        success_count += 1
                    except Exception as e:
                        st.error(f"Error adding DDL statement: {str(e)}")

                if success_count > 0:
                    st.success(f"Successfully added {success_count} DDL statements")
                    # Refresh training data
                    st.session_state["training_data"] = get_training_data()
                else:
                    st.error("Failed to add any DDL statements")

        except json.JSONDecodeError:
            st.error("Invalid JSON file. Please check the file format.")
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

    # Display Training Data
    st.subheader("Training Data")
    if st.button("Refresh Training Data"):
        st.session_state["training_data"] = get_training_data()

    # Add custom CSS for red buttons
    st.markdown("""
        <style>
            .red-button {
                background-color: #ff4d4d;
                color: white;
                border: none;
                padding: 0.5em 1em;
                border-radius: 0.25em;
                font-size: 1em;
                cursor: pointer;
                margin: 0.25em 0;
            }
        </style>
    """, unsafe_allow_html=True)

    # Use st.columns to create space for each button
    col1, col2, col3, col4, _, _, _, _ = st.columns(8)

    with col1:
        if st.button("Delete All", key="delete_all_training_data"):
            remove_collection_cached("sql")
            remove_collection_cached("ddl")
            remove_collection_cached("documentation")
            st.session_state["training_data"] = get_training_data()

    with col2:
        if st.button("Delete SQLs", key="delete_sql"):
            remove_collection_cached("sql")
            st.session_state["training_data"] = get_training_data()

    with col3:
        if st.button("Delete DDLs", key="delete_ddl"):
            remove_collection_cached("ddl")
            st.session_state["training_data"] = get_training_data()

    with col4:
        if st.button("Delete Documentation", key="delete_documentation"):
            remove_collection_cached("documentation")
            st.session_state["training_data"] = get_training_data()

    # Initialize or get training data from session state
    if "training_data" not in st.session_state:
        st.session_state["training_data"] = get_training_data()

    training_data = st.session_state["training_data"]

    # Display training data with delete buttons
    if not training_data.empty:
        # Add a delete button column
        for i, row in training_data.iterrows():
            col1, col2 = st.columns([4, 1])
            with col1:
                expander = st.expander(f"{row['training_data_type'].upper()}: {row['id']}")
                if row['question']:
                    expander.markdown(f"**Question:** {row['question']}")
                expander.markdown(f"**Content:**\n```\n{row['content']}\n```")
            with col2:
                if st.button("Delete", key=f"delete_{row['id']}"):
                    if remove_training_data_cached(id=row['id']):
                        st.success(f"Deleted {row['id']}")
                        # Refresh training data
                        st.session_state["training_data"] = get_training_data()
    else:
        st.info("No training data available.")
