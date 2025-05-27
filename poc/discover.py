import pandas as pd
from sqlalchemy import create_engine, inspect
from sqlalchemy_utils import database_exists
from sqlalchemy.engine.url import URL
from sqlalchemy.exc import SQLAlchemyError
from urllib.parse import quote_plus

import json
import streamlit as st

from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import AzureChatOpenAI

import os
import io
import csv 
from concurrent.futures import ThreadPoolExecutor


def create_db_engine(config):
    drivers = {
        "Microsoft SQL Server": "mssql+pyodbc",
        "MySQL": "mysql+pymysql",
        "OracleDB": "oracle+cx_oracle",
        "SQLite": "sqlite",
        "PostgreSQL": "postgresql",
    }

    if 'password' in config:
        config['password'] = quote_plus(config['password'])
    
    drivername = drivers.get(config['rdbms'])
    if not drivername:
        return None, f"Unsupported RDBMS: {config['rdbms']}"
    
    try:
        if config['rdbms'] == "Microsoft SQL Server":
            if config['authentication'] == "Windows Authentication":
                engine = create_engine(f"{drivername}://{config['host']}/{config['database']}?driver=ODBC+Driver+17+for+SQL+Server&Trusted_Connection=yes")
            else:
                engine = create_engine(f"{drivername}://{config['username']}:{config['password']}@{config['host']}/{config['database']}?driver=ODBC+Driver+17+for+SQL+Server")
        elif config['rdbms'] == "OracleDB":
            engine = create_engine(
                f"{drivername}://{config['username']}:{config['password']}@{config['host']}:{config['port']}/{config['service']}",
                connect_args={"encoding": "UTF-8", "nencoding": "UTF-8"}
            )
        elif config['rdbms'] == "SQLite":
            config['database'] = config['database'].strip('"').replace('\\','/')
            if database_exists(f"{drivername}:///{config['database']}"):
                engine = create_engine(f"{drivername}:///{config['database']}")
        else:
            engine = create_engine(URL.create(
                drivername=drivername,
                username=config.get('username'),
                password=config.get('password'),
                host=config.get('host'),
                port=config.get('port'),
                database=config.get('database')
            ))
        return engine, None
    except SQLAlchemyError as e:
        return None, f"Connection failed: {e}"

def get_schemas(engine):
    try:
        if engine.name == 'sqlite':
            return ['main'], None
        
        inspector = inspect(engine)
        schemas = inspector.get_schema_names()

        system_schemas = ['INFORMATION_SCHEMA', 'sys',
                          'information_schema', 'mysql', 'performance_schema',
                          'SYS', 'SYSTEM', 'OUTLN', 'CTXSYS', 'MDSYS']
        
        schemas = [schema for schema in schemas if 
                   schema not in system_schemas 
                   and not schema.startswith('db_') 
                   and not schema.startswith('pg_')]
        
        return schemas, None
    except SQLAlchemyError as e:
        return None, f"Error fetching schemas: {e}"

def get_tables(engine, schema=None):
    try:
        inspector = inspect(engine)
        if engine.name == 'sqlite':
            return inspector.get_table_names(), None
        return inspector.get_table_names(schema=schema), None
    except SQLAlchemyError as e:
        return None, f"Error fetching tables for schema {schema}: {e}"

def fetch_table_data(engine, table_name, schema):
    try:
        if engine.name == 'mssql':
            query = f"SELECT TOP 5 * FROM {schema}.{table_name}"
        elif engine.name == 'oracle':
            query = f"SELECT * FROM {schema}.{table_name} FETCH FIRST 5 ROWS ONLY;"
        elif engine.name == 'sqlite':
            query = f"SELECT * FROM {table_name} LIMIT 5"
        else:
            query = f"SELECT * FROM {schema}.{table_name} LIMIT 5"  
        df = pd.read_sql(query, engine)
        return df
    except SQLAlchemyError as e:
        return e
    
def fetch_table_metadata(engine, table_name, schema=None):
    try:
        inspector = inspect(engine)
        columns = inspector.get_columns(table_name, schema=schema)
        pk_constraint = inspector.get_pk_constraint(table_name, schema=schema)
        foreign_keys = inspector.get_foreign_keys(table_name, schema=schema)
        if engine.name == 'mssql':
            unique_constraints = [index for index in inspector.get_indexes(table_name, schema=schema) if index['unique']]
        else:
            unique_constraints = inspector.get_unique_constraints(table_name, schema=schema)
        
        # Convert SQLAlchemy types to Python native types for JSON serialization
        for col in columns:
            col['type'] = str(col['type'])
            # Extract NULL, DEFAULT, and NOT NULL constraints
            col['nullable'] = col.get('nullable', True)
            col['default'] = str(col.get('default', ''))
            col['constraints'] = 'NOT NULL' if not col['nullable'] else 'NULL'
            if col['default']:
                col['constraints'] += f", DEFAULT {col['default']}"
        
        metadata = {
            "columns": columns,
            "primary_key": pk_constraint or {},
            "foreign_keys": foreign_keys,
            "unique_constraints": unique_constraints
        }
        return metadata, None
    except SQLAlchemyError as e:
        return None, f"Error fetching metadata for table {table_name}: {e}"

def create_table_overview(connections, selected_schemas_per_connection, selected_tables_per_schema_per_connection):
    overview = {connection_key: [] for connection_key in connections.keys()}

    for connection_key, connection_info in connections.items():
        engine = connection_info['engine']
        for schema in selected_schemas_per_connection[connection_key]:
            selected_tables = selected_tables_per_schema_per_connection[schema]
            for table in selected_tables:
                metadata, error = fetch_table_metadata(engine, table, schema=schema)
                # print(metadata)
                if error:
                    print(error)
                    continue
                
                # Format metadata for display, including constraints
                formatted_metadata = []
                for col in metadata['columns']:
                    column_info = f"{col['name']}: ({col['type']})"
                    constraints_info = f" [{col['constraints']}]"
                    if col['name'] in (metadata['primary_key'].get('constrained_columns') or []):
                        column_info += " [Primary Key]"
                    for fk in metadata['foreign_keys']:
                        if col['name'] in fk['constrained_columns']:
                            column_info += f" [Foreign Key references {fk['referred_table']}.{fk['referred_columns'][0]}]"
                    for uc in metadata['unique_constraints']:
                        if col['name'] in uc['column_names']:
                            column_info += " [Unique]"
                    column_info += constraints_info
                    formatted_metadata.append(column_info)
                
                table_info = {
                    "Schema": schema,
                    "Table": table,
                    "Metadata": formatted_metadata
                }
                # Append table information to the overview under the connection key
                overview[connection_key].append(table_info)
    
    return overview

def chunk_schema(schema_data):
    schema_batches = []
    for tablespace, schema in schema_data.items():
        schema_info_chunks = []
        current_chunk = f"Tablespace: {tablespace}\n"
        for item in schema:
            table_info = f"Table: {item["Schema"]}.{item["Table"]}\nMetadata: {item["Metadata"]}\n"
            current_chunk += table_info
            schema_info_chunks.append(current_chunk)
            current_chunk = ""
        schema_batches.append(schema_info_chunks)
        print(f"{schema_info_chunks}\n\n")
    return schema_batches


# db_schema_prompt = PromptTemplate.from_template(
# """
# You are a data analyst with expertise in multiple domains including ESG, BFSI, Tax, Strategy and Transactions. 
# Create a comprehensive business glossary from the given database information. Reason step-by-step based on the following 
# instruction:
# 1. Start with the tablespace name (if provided inside schema).
# 2. For each table, generate an informative description by drawing context from the name and metadata of the table in the schema data.
# 3. List all columns for each table with comprehensive and business-centric definitions.
# 4. Infer meanings for incomplete or ambiguous names based on context. 
# 5. If uncertain about a meaning, use 'not enough context'.
# 6. The hierarchy is such that each tablespace can contain multiple schemas and each schema can contain multiple tables. 
# Ensure tablespace name is not redundantly included.
# 7. Generate a complete glossary for every table.

# Database Information: {schema_definition}

# Use this format throughout:
# Tablespace: [Tablespace Name]
# Schema: [Schema Name]
# Table: [Table Name]
# [Concise description of the table]
                                
# [Column Name] | [Comprehensive and business-centric definition for the column]  
                                                
# Do not include introductory or closing messages.
# """)

db_schema_prompt = PromptTemplate.from_template(
"""
You are a business analyst and subject matter expertise in multiple domains of enterprise including ESG, BFSI and retail. 
You will create a comprehensive business glossary from the given database information by reasoning step-by-step 
based on the following instruction:
1. For each table, generate an informative description by drawing context from the name and metadata of the table in 
the schema data.
2. List all columns for each table with comprehensive and business-centric definitions. Industry specific terminologies
should be elaborated upon such that the reader finds the glossary useful for data engineering as well as business analytics.
3. Infer meanings for incomplete or ambiguous names based on context. 

Table Info: 
{schema_data}

Use this format throughout:
Tablespace: [Tablespace name if provided in Table Info]
Table: [Table Name]
[Concise description of the table]
                                
[Column 1 Name] | [Comprehensive and business-centric definition for the column]  
[Column 2 Name] | [Comprehensive and business-centric definition for the column] 
...
[Column 3 Name] | [Comprehensive and business-centric definition for the column] 
-------
                                                
Do not include introductory or closing messages. After describing all columns, end your response with ------.
""")

db_schema_continue_prompt = PromptTemplate.from_template(
"""
You are a business analyst and subject matter expertise in multiple domains of enterprise including ESG, BFSI and retail. 
You created part of a comprehensive business glossary from the given database information by reasoning step-by-step 
based on the following instruction:
1. For each table, generate an informative description by drawing context from the name and metadata of the table in 
the schema data.
2. List all columns for each table with comprehensive and business-centric definitions. Industry specific terminologies,
such as metrics and instruments should be elaborated upon such that the reader finds the glossary useful for data 
engineering as well as business analytics.
3. Infer meanings for incomplete or ambiguous names based on context. 

Table Info: 
{schema_data}

Glossary generated so far:
{glossary}

Use this format throughout:
Tablespace: [Tablespace name if provided in Table Info]
Table: [Table Name]
[Concise description of the table]
                                
[Column 1 Name] | [Comprehensive and business-centric definition for the column]  
[Column 2 Name] | [Comprehensive and business-centric definition for the column] 
...
[Column 3 Name] | [Comprehensive and business-centric definition for the column] 
------
                                                
Do not include introductory or closing messages. After describing all columns, end your response with ------.
""")


# def create_glossary(schema_data):
#     llm = AzureChatOpenAI(
#         azure_deployment="gpt-4o",
#         openai_api_version="2024-02-15-preview",
#         api_key="901cce4a4b4045b39727bf1ce0e36219",
#         azure_endpoint="https://aifordata-azureopenai.openai.azure.com/",
#         temperature=0,
#         max_tokens=4096
#     )

#     runnable = RunnablePassthrough()
#     output_parser = StrOutputParser()
#     chain = runnable | db_schema_prompt | llm | output_parser

#     response = chain.invoke({'schema_definition': {schema_data}})

#     # print(response)

#     # Return the combined formatted glossary and CSV data
#     return format_glossary(response)

def generate_glossary(chunk):
    llm = AzureChatOpenAI(
        azure_deployment="gpt-4o",
        openai_api_version="2024-02-15-preview",
        api_key="901cce4a4b4045b39727bf1ce0e36219",
        azure_endpoint="https://aifordata-azureopenai.openai.azure.com/",
        temperature=0,
        max_tokens=4096
    )

    runnable = RunnablePassthrough()
    output_parser = StrOutputParser()

    chain = runnable | db_schema_prompt | llm | output_parser
    continue_chain = runnable | db_schema_continue_prompt | llm | output_parser

    glossary = chain.invoke({"schema_data": chunk})

    while not glossary.endswith("------"):
        glossary_lines = glossary.split('\n')
        # chunk_lines = chunk.split('\n')
        # if chunk_lines[-1].split(":")[0] in glossary_lines[-1]:
        #     glossary = '\n'.join(glossary_lines)
        #     break
        glossary = '\n'.join(glossary_lines[:-1])
        continued_glossary = continue_chain.invoke({"schema_data": chunk, "glossary": glossary})
        glossary += f"\n{continued_glossary}"

    return glossary

def create_glossary(schema_data):
    schema_data_dict = json.loads(schema_data)
    schema_batches = chunk_schema(schema_data_dict)
    glossaries = []
    for batch in schema_batches:
        with ThreadPoolExecutor() as executor:
            futures = []
            for chunk in batch:
                future = executor.submit(generate_glossary, chunk)
                futures.append(future)

        glossaries.extend([future.result() for future in futures])

        # if not isinstance(glossaries, list):
        #     raise TypeError("Expected glossaries to be a list of strings")
        
        # for i in range(len(glossaries)):
        #     start_index = glossaries[i].find("Tablespace:")
        #     if start_index != -1:
        #         glossaries[i] = glossaries[i][start_index:]

    glossary = "  \n".join(glossaries)
    return format_glossary(glossary)

def format_glossary(response):
    lines = response.strip().split('\n')
    formatted_output = []
    csv_data = []
    
    current_tablespace = None
    current_table = None
    for line in lines:
        if line.strip() and '|' not in line:
            if "Tablespace:" in line: # This is a tablespace name
                current_tablespace = line.strip()
                formatted_output.append(f"\n### {current_tablespace}")
                current_tablespace = current_tablespace.replace("Tablespace:", "")
            # elif "Schema: " in line:
            #     current_schema = line.strip()
            #     formatted_output.append(f"\n#### {current_schema}")
            #     current_tablespace = current_schema.replace("Schema:", "")    
            elif "Table: " in line:  # This is a table name
                current_table = line.strip()
                formatted_output.append(f"\n#### {current_table}")
                current_table = current_table.replace("Table:", "")            
            else:
                current_table_description = line.strip()
                formatted_output.append(f"{current_table_description}")
        elif '|' in line:
            parts = line.split('|')
            if len(parts) == 2:
                term, definition = parts
                if term.strip().lower() != "term":
                    formatted_output.append(f"- **{term.strip()}**: {definition.strip()}")
                    # csv_data.append([current_tablespace, current_schema, current_table, current_table_description, f"{term.strip()}", definition.strip()])
                    csv_data.append([current_tablespace, current_table, current_table_description, f"{term.strip()}", definition.strip()])
            else:
                return None, None
            
    # print(csv_data)
    
    return '\n'.join(formatted_output), csv_data

def generate_csv(csv_data):
    output = io.StringIO()
    writer = csv.writer(output)
    # writer.writerow(['Tablespace', 'Schema', 'Table', 'Table Description', 'Column', 'Definition'])
    writer.writerow(['Tablespace', 'Table', 'Table Description', 'Column', 'Definition'])
    writer.writerows(csv_data)
    return output.getvalue()


# Streamlit interface
st.set_page_config(
    layout="wide",
    page_title="Discover and Preview from Data Sources",
    page_icon="🔍",
)

st.subheader("Discover Metadata and Preview Data from Multiple Data Sources across various RDBMS Technologies 🔍")
st.markdown(
"""
This tool can connect to SQLite, Microsoft SQL Server, PostgreSQL, OracleDB and MySQL databases. You may pick specific schemas 
and tables, view their metadata and sample data and generate a business glossary for your selections.

----
""")


# Initialize session state
if 'connections' not in st.session_state:
    st.session_state['connections'] = {}
if 'selected_tables_per_schema_per_connection' not in st.session_state:
    st.session_state['selected_tables_per_schema_per_connection'] = {}
if 'selected_schemas_per_connection' not in st.session_state:
    st.session_state['selected_schemas_per_connection'] = {}
if 'selected_connection' not in st.session_state:
    st.session_state['selected_connection'] = None
if 'tablespace_overview' not in st.session_state:
    st.session_state['tablespace_overview'] = None
if 'preview_data' not in st.session_state:
    st.session_state['preview_data'] = None


# Sidebar for connection setup and selection
with st.sidebar:
    with st.expander("Create Connections"):
        db_type = st.selectbox("Database Type", ["PostgreSQL", "MySQL", "Microsoft SQL Server", "OracleDB", "SQLite"])
        
        connection_details = {}
        if db_type == "SQLite":
            connection_details['database'] = st.text_input("Database File Path")
        else:
            connection_details['host'] = st.text_input("Host", type="password")
            connection_details['username'] = st.text_input("Username", type="password")
            connection_details['password'] = st.text_input("Password", type="password")
            if db_type != "OracleDB":
                connection_details['database'] = st.text_input("Database Name", type="password")
            if db_type in ["PostgreSQL", "MySQL"]:
                connection_details['port'] = st.text_input("Port")
            if db_type == "OracleDB":
                connection_details['service'] = st.text_input("Service Name")
            if db_type == "Microsoft SQL Server":
                connection_details['authentication'] = st.selectbox("Authentication Type", ["Windows Authentication", "Others"])
        
        if st.button("Add Data Source"):
            config = {
                'rdbms': db_type,
                **connection_details
            }
            
            with st.spinner("Testing connection..."):
                engine, error = create_db_engine(config)
            
            if engine:
                connection_key = f"{db_type} > {connection_details.get('database', '')}"
                schemas, schema_error = get_schemas(engine)
                if schema_error:
                    st.error(schema_error)
                else:
                    st.session_state['connections'][connection_key] = {
                        'engine': engine,
                        'schemas': schemas,
                        'type': db_type
                    }
                    st.session_state['selected_connection'] = connection_key
                    st.success("Connection added successfully!")
            else:
                st.error(error)

    # Select Data Source and Schema
    if st.session_state['connections']:
        connection_keys = list(st.session_state['connections'].keys())
        selected_connection = st.selectbox(
            "Select Database Connection",
            connection_keys,
            index=connection_keys.index(st.session_state['selected_connection']) if st.session_state['selected_connection'] in connection_keys else 0
        )
        st.session_state['selected_connection'] = selected_connection
        
        if selected_connection:
            connection_info = st.session_state['connections'][selected_connection]
            engine = connection_info['engine']
            schemas = connection_info['schemas']
            db_type = connection_info['type']
            
            if db_type == "SQLite":
                selected_schemas = ['main']
                st.info("SQLite does not support multiple schemas. Using 'main' schema.")
            else:
                selected_schemas = st.multiselect("Select Schemas", schemas)

            st.session_state['selected_schemas_per_connection'][selected_connection] = selected_schemas
            
            for schema in selected_schemas:
                tables, table_error = get_tables(engine, schema=None if db_type == "SQLite" else schema)
                if table_error:
                    st.error(table_error)
                else:
                    selected_tables = st.session_state['selected_tables_per_schema_per_connection'].get(schema, [])
                    new_selected_tables = st.multiselect(f"Choose tables for schema: {schema}", tables, default=selected_tables, key=schema)
                    st.session_state['selected_tables_per_schema_per_connection'][schema] = new_selected_tables
            
            # selected_tables = st.session_state['selected_tables_per_connection'].get(selected_connection, [])
            # new_selected_tables = st.multiselect("Choose tables", all_tables, default=selected_tables)
            # st.session_state['selected_tables_per_connection'][selected_connection] = new_selected_tables


# Tablespace Overview Section
if st.session_state['selected_connection']:
    st.subheader("Tablespace Overview", divider='rainbow')
    overview = create_table_overview(
        st.session_state['connections'],
        st.session_state['selected_schemas_per_connection'],
        st.session_state['selected_tables_per_schema_per_connection']
    )
    st.session_state['tablespace_overview'] = overview
    
    for connection_key, tables in overview.items():
        with st.expander(f"{connection_key}"):
            for item in tables:
                # Initialize checkbox state key
                checkbox_key = f"metadata-{connection_key} > {item['Schema']} > {item['Table']}"
                if checkbox_key not in st.session_state:
                    st.session_state[checkbox_key] = False  # Default to False to not affect sidebar selection
                
                # Create checkbox and use the state
                table_selected = st.checkbox(f"{item['Schema']} > {item['Table']}", key=checkbox_key, value=st.session_state[checkbox_key])
                if table_selected:
                    st.write(f"**Metadata:**")
                    for column_info in item['Metadata']:
                        st.write(f"- {column_info}")

                    if st.button(f"Preview {item['Table']}", key=f"preview-{connection_key} > {item['Schema']} > {item['Table']}"):
                        # Fetch table data for preview
                        with st.spinner(f"Fetching data for {item['Table']}..."):
                            preview_data = fetch_table_data(
                                st.session_state['connections'][connection_key]['engine'],
                                item['Table'],
                                item['Schema']
                            )
                            st.dataframe(preview_data)

# Business Glossary Section
if st.session_state['tablespace_overview']:
    if st.button("Generate Business Glossary"):
        with st.spinner("Generating Business Glossary..."):
            selected_tables_with_metadata_for_glossary = {}
            for connection_key, tables in st.session_state['tablespace_overview'].items():
                selected_tables_metadata = [
                    {"Schema": item['Schema'], "Table": item['Table'], "Metadata": item['Metadata']} for item in tables
                    if st.session_state[f"metadata-{connection_key} > {item['Schema']} > {item['Table']}"]
                ]
                if selected_tables_metadata:
                    selected_tables_with_metadata_for_glossary[connection_key] = selected_tables_metadata
            
            if selected_tables_with_metadata_for_glossary:
                st.subheader("Business Glossary", divider='rainbow')
                schema_data_with_metadata = json.dumps(selected_tables_with_metadata_for_glossary)
                formatted_glossary, csv_data = create_glossary(schema_data_with_metadata)
                st.markdown(formatted_glossary)
                
                csv_content = generate_csv(csv_data)
                st.download_button(
                    label="Download CSV",
                    data=csv_content,
                    file_name="Business Glossary.csv",
                    mime="text/csv"
                )
            else:
                st.warning("No tables have been selected for the Business Glossary.")
