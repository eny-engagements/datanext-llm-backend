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

from databricks import sql as databricks_sql

def create_db_engine(config):
    drivers = {
        "Microsoft SQL Server": "mssql+pyodbc",
        "MySQL": "mysql+pymysql",
        "OracleDB": "oracle+cx_oracle",
        "SQLite": "sqlite",
        "PostgreSQL": "postgresql",
        "Databricks" : "databricks"
    }

    if 'password' in config:
        config['password'] = quote_plus(config['password'])
    
    drivername = drivers.get(config['rdbms'])
    if not drivername:
        return None, f"Unsupported RDBMS: {config['rdbms']}"
    
    try:
        if config['rdbms'] == "Databricks":          
            engine = databricks_sql.connect(
                server_hostname=config["server_hostname"],
                http_path=config["http_path"],
                access_token=config["access_token"]
            )
            # print("Databricks Connection Done")
        
        elif config['rdbms'] == "Microsoft SQL Server":
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

def get_catalogs_databricks(connection):
    try:
        cursor = connection.cursor()
        cursor.execute("SHOW CATALOGS")
        result = [row[0] for row in cursor.fetchall()]
        cursor.close()
        return result, None
    except Exception as e:
        return None, f"Error fetching catalogs: {e}"

def get_schemas_databricks(connection, catalog):
    try:
        cursor = connection.cursor()
        cursor.execute(f"SHOW SCHEMAS IN {catalog}")
        result = [row[0] for row in cursor.fetchall()]
        cursor.close()
        return result, None
    except Exception as e:
        return None, f"Error fetching schemas in catalog {catalog}: {e}"

def get_tables_databricks(connection, catalog, schema):
    try:
        cursor = connection.cursor()
        cursor.execute(f"SHOW TABLES IN {catalog}.{schema}")
        result = [row[1] for row in cursor.fetchall()]  # row[1] is table name
        cursor.close()
        return result, None
    except Exception as e:
        return None, f"Error fetching tables in {catalog}.{schema}: {e}"

def get_schemas(engine):
    try:
        if engine.name == 'sqlite':
            return ['main'], None
        
        inspector = inspect(engine)
        schemas = inspector.get_schema_names()

        system_schemas = ['INFORMATION_SCHEMA', 'sys', 'information_schema', 'mysql', 'performance_schema',
                          'SYS', 'SYSTEM', 'OUTLN', 'CTXSYS', 'MDSYS']
        schemas = [schema for schema in schemas if schema not in system_schemas 
                   and not schema.startswith('db_') and not schema.startswith('pg_')]

        return schemas, None
    except Exception as e:
        return None, f"Error fetching schemas: {e}"

def get_tables(engine, schema=None):
    try:
        if hasattr(engine, "cursor"):  # Databricks
            with engine.cursor() as cursor:
                cursor.execute(f"""
                    SELECT table_name 
                    FROM system.information_schema.tables 
                    WHERE table_schema = '{schema}'
                """)
                rows = cursor.fetchall()
                return [row[0] for row in rows], None

        inspector = inspect(engine)
        if engine.name == 'sqlite':
            return inspector.get_table_names(), None
        return inspector.get_table_names(schema=schema), None
    except Exception as e:
        return None, f"Error fetching tables for schema {schema}: {e}"



def fetch_table_data(engine, table_name, schema=None, catalog=None, db_type=None):
    try:
        if db_type == "Databricks":
            fq_table = f"{catalog}.{schema}.{table_name}"
            query = f"SELECT * FROM {fq_table} LIMIT 5"
            cursor = engine.cursor()
            cursor.execute(query)
            rows = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            df = pd.DataFrame(rows, columns=columns)
            cursor.close()
            return df
        else:
            if engine.name == 'mssql':
                query = f"SELECT TOP 5 * FROM {schema}.{table_name}"
            elif engine.name == 'oracle':
                query = f"SELECT * FROM {schema}.{table_name} FETCH FIRST 5 ROWS ONLY"
            elif engine.name == 'sqlite':
                query = f"SELECT * FROM {table_name} LIMIT 5"
            else:
                query = f"SELECT * FROM {schema}.{table_name} LIMIT 5"
            df = pd.read_sql(query, engine)
            return df
    except Exception as e:
        return e
    
def fetch_table_metadata(engine, table_name, schema=None, catalog=None, db_type=None):
    try:
        if db_type == "Databricks":
            cursor = engine.cursor()

            # 1. Column metadata
            cursor.execute(f"""
                SELECT column_name, data_type, is_nullable
                FROM {catalog}.information_schema.columns
                WHERE table_schema = '{schema}' AND table_name = '{table_name}'
            """)
            columns = []
            for row in cursor.fetchall():
                col_name, col_type, is_nullable = row
                col_info = {
                    "name": col_name,
                    "type": col_type,
                    "nullable": is_nullable == 'YES',
                    "default": "",
                    "constraints": "NULL" if is_nullable == 'YES' else "NOT NULL"
                }
                columns.append(col_info)

            # 2. Primary keys
            cursor.execute(f"""
                SELECT column_name
                FROM {catalog}.information_schema.table_constraints tc
                JOIN {catalog}.information_schema.key_column_usage kcu
                  ON tc.constraint_name = kcu.constraint_name
                WHERE tc.constraint_type = 'PRIMARY KEY'
                  AND tc.table_schema = '{schema}'
                  AND tc.table_name = '{table_name}'
            """)
            primary_key_columns = [row[0] for row in cursor.fetchall()]

            # 3. Foreign keys
            cursor.execute(f"""
                SELECT column_name
                FROM {catalog}.information_schema.table_constraints tc
                JOIN {catalog}.information_schema.key_column_usage kcu
                ON tc.constraint_name = kcu.constraint_name
                WHERE tc.constraint_type = 'FOREIGN KEY'
                AND tc.table_schema = '{schema}'
                AND tc.table_name = '{table_name}'
            """)
            foreign_keys = [
                {
                    "constrained_columns": [row[0]],
                    "referred_table": None,
                    "referred_columns": []
                }
                for row in cursor.fetchall()
            ]

            # 4. Apply constraints
            for col in columns:
                if col["name"] in primary_key_columns:
                    col["constraints"] += ", PRIMARY KEY"

            metadata = {
                "columns": columns,
                "primary_key": {"constrained_columns": primary_key_columns},
                "foreign_keys": foreign_keys,
                "unique_constraints": []  # can be added later
            }

            cursor.close()
            return metadata, None

        # For non-Databricks engines, use SQLAlchemy inspection
        inspector = inspect(engine)
        columns = inspector.get_columns(table_name, schema=schema)
        pk_constraint = inspector.get_pk_constraint(table_name, schema=schema)
        foreign_keys = inspector.get_foreign_keys(table_name, schema=schema)

        if engine.name == 'mssql':
            unique_constraints = [index for index in inspector.get_indexes(table_name, schema=schema) if index['unique']]
        else:
            unique_constraints = inspector.get_unique_constraints(table_name, schema=schema)

        for col in columns:
            col['type'] = str(col['type'])
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

    except Exception as e:
        return None, f"Error fetching metadata for table {table_name}: {e}"

def create_table_overview(connections, selected_schemas_per_connection, selected_tables_per_schema_per_connection, catalog_lookup, db_type_lookup):
    overview = {connection_key: [] for connection_key in connections.keys()}

    for connection_key, connection_info in connections.items():
        engine = connection_info['engine']
        for schema in selected_schemas_per_connection[connection_key]:
            selected_tables = selected_tables_per_schema_per_connection[schema]
            catalog = catalog_lookup.get(connection_key)
            db_type = db_type_lookup.get(connection_key)
            for table in selected_tables:
                metadata, error = fetch_table_metadata(engine, table, schema=schema ,catalog=catalog, db_type=db_type)
                # print(metadata)
                if error:
                    # print(error)
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

# def chunk_schema(schema_data):
#     schema_batches = []
#     for tablespace, schema in schema_data.items():
#         schema_info_chunks = []
#         current_chunk = f"Tablespace: {tablespace}\n"
#         for item in schema:
#             table_info = f"Table: {item["Schema"]}.{item["Table"]}\nMetadata: {item["Metadata"]}\n"
#             current_chunk += table_info
#             schema_info_chunks.append(current_chunk)
#             current_chunk = ""
#         schema_batches.append(schema_info_chunks)
#         # print(f"{schema_info_chunks}\n\n")
#     return schema_batches

def chunk_schema(schema_data):

    schema_batches = []

    for tablespace, table_list in schema_data.items():
        # table_list is a list of dicts like { "Schema": "...", "Table": "...", "Metadata": […] }

        single_connection_chunks = []
        for item in table_list:
            # Always start a brand-new chunk with the “Tablespace:” line
            chunk = f"Tablespace: {tablespace}\n"
            chunk += f"Table: {item['Schema']}.{item['Table']}\n"
            chunk += f"Metadata: {item['Metadata']}\n"

            single_connection_chunks.append(chunk)

        schema_batches.append(single_connection_chunks)

    return schema_batches

def update_unity(df, selected_connection, session_state):
    """
    Applies table-level and column-level comments to Unity Catalog using the Databricks SQL connector.
    """
    try:
        engine = session_state['connections'][selected_connection]['engine']
        catalog = session_state['selected_catalog']
        schema_list = session_state['selected_schemas_per_connection'][selected_connection]

        success_count = 0
        failed_comments = []
        updated_tables = set()  # track tables already updated with table-level description

        cursor = engine.cursor()

        for _, row in df.iterrows():
            schema = None

            # Detect schema from fully qualified table name
            for possible_schema in schema_list:
                if row['Table'].startswith(possible_schema + "."):
                    schema = possible_schema
                    # print("schema", schema)
                    table_name = row['Table'].split(".")[1]
                    break

            if not schema:
                schema = schema_list[0]  # fallback
                # print("schema ###", schema)
                table_name = row['Table']
                # print("table name" , table_name)

            column_name = row['Column']
            comment_text = row['Definition'].replace("'", "")  # escape single quotes
            table_description = row['Table Description'].replace("'", "")  # escape quotes

            fq_table = f"{catalog}.{table_name}"
            # print("table path" , fq_table)

            # Step 1: Set table-level comment (only once per table)
            if fq_table not in updated_tables:
                try:
                    table_comment_query = f"""COMMENT ON TABLE {fq_table} IS '{table_description}'"""
                    # print(f"[TABLE] {table_comment_query}")
                    cursor.execute(table_comment_query)
                    updated_tables.add(fq_table)
                except Exception as e:
                    failed_comments.append((table_name, "[TABLE_DESC]", str(e)))

            # Step 2: Set column-level comment
            try:
                col_comment_query = f"""COMMENT ON COLUMN {fq_table}.{column_name} IS '{comment_text}'"""
                # print(f"[COLUMN] {col_comment_query}")
                cursor.execute(col_comment_query)
                success_count += 1
            except Exception as e:
                failed_comments.append((table_name, column_name, str(e)))

        cursor.close()
        return success_count, failed_comments

    except Exception as e:
        return 0, [("GENERAL_ERROR", "N/A", str(e))]

    except Exception as e:
        print('Error')
        return 0, [("GENERAL_ERROR", "N/A", str(e))]
    
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
    print("chunk:", chunk)
    print("======================================LLM==========")
    print("glossary:", glossary)
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
                    # print(csv_data)
            else:
                return None, None           
    
    return '\n'.join(formatted_output), csv_data

def generate_csv(csv_data):
    output = io.StringIO()
    writer = csv.writer(output)
    # writer.writerow(['Tablespace', 'Schema', 'Table', 'Table Description', 'Column', 'Definition'])
    writer.writerow(['Tablespace', 'Table', 'Table Description', 'Column', 'Definition'])
    writer.writerows(csv_data)
    return output.getvalue()


##Convert the glossary into df
def generate_df (csv_data):
    headers = ['Tablespace', 'Table', 'Table Description', 'Column', 'Definition']
    df = pd.DataFrame(csv_data, columns=headers)
    return df

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
        db_type = st.selectbox("Database Type", ["PostgreSQL", "MySQL", "Microsoft SQL Server", "OracleDB", "SQLite", "Databricks"])
        
        connection_details = {}
        
        if db_type == "Databricks":
            connection_details['server_hostname'] = st.text_input("Server Hostname (e.g. adb-xxx.azuredatabricks.net)")
            connection_details['http_path'] = st.text_input("HTTP Path (from SQL warehouse)")
            connection_details['access_token'] = st.text_input("Access Token", type="password")

        elif db_type == "SQLite":
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
                # print(engine)
            
            if db_type == "Databricks":
                connection_key = f"{db_type} > {connection_details.get('server_hostname')}"
                st.session_state['connections'][connection_key] = {
                    'engine':engine,
                    'schemas':[],
                    'type':db_type
                }
                st.session_state['selected_connection'] = connection_key
                st.success("Databrick Connection Added.")
            
            elif engine:
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
            
            if db_type == "Databricks":
                catalogs, catalog_error = get_catalogs_databricks(engine)
                if catalog_error:
                    st.error(catalog_error)
                else:
                    selected_catalog = st.selectbox("Select Catalog", catalogs)
                    st.session_state['selected_catalog'] = selected_catalog
                    schemas, schema_error = get_schemas_databricks(engine, selected_catalog)
                    if schema_error:
                        st.error(schema_error)
                    else:
                        selected_schemas = st.multiselect("Select Schemas", schemas)
                        st.session_state['selected_schemas_per_connection'][selected_connection] = selected_schemas
                        if db_type not in ['Databricks', 'SQLite']:
                            for schema in selected_schemas:
                                tables, table_error = get_tables_databricks(engine, selected_catalog, schema)
                                if table_error:
                                    st.error(table_error)
                                else:
                                    selected_tables = st.session_state['selected_tables_per_schema_per_connection'].get(schema, [])
                                    new_selected_tables = st.multiselect(
                                        f"Choose tables for {selected_catalog}.{schema}",
                                        tables,
                                        default=selected_tables,
                                        key=f"{selected_catalog}.{schema}"
                                    )
                                    st.session_state['selected_tables_per_schema_per_connection'][schema] = new_selected_tables

            elif db_type == "SQLite":
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
        st.session_state['selected_tables_per_schema_per_connection'],
        catalog_lookup = {selected_connection: st.session_state.get('selected_catalog')},
        db_type_lookup = {selected_connection: db_type}
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
                            catalog = st.session_state.get('selected_catalog')
                            preview_data = fetch_table_data(
                                st.session_state['connections'][connection_key]['engine'],
                                item['Table'],
                                item['Schema'],
                                catalog=catalog,
                                db_type=db_type
                            )
                            st.dataframe(preview_data)

glossary_exists = "business_glossary_df" in st.session_state
# Business Glossary Section
if st.session_state['tablespace_overview']:
        if st.button("Generate Business Glossary", key="generate_glossary"):
            # Build a dict of only those tables that have their metadata‐checkbox = True
            selected_tables_with_metadata_for_glossary = {}
            for conn_key, tables in st.session_state["tablespace_overview"].items():
                selected_tables_metadata = [
                    {
                        "Schema": item["Schema"],
                        "Table": item["Table"],
                        "Metadata": item["Metadata"]
                    }
                    for item in tables
                    if st.session_state.get(f"metadata-{conn_key} > {item['Schema']} > {item['Table']}", False)
                ]
                if selected_tables_metadata:
                    selected_tables_with_metadata_for_glossary[conn_key] = selected_tables_metadata

            if not selected_tables_with_metadata_for_glossary:
                st.warning("No tables have been selected for the Business Glossary.")
            else:
                with st.spinner("Generating Business Glossary…"):
                    # Convert to JSON string and call create_glossary(...)
                    schema_data_with_metadata = json.dumps(selected_tables_with_metadata_for_glossary)
                    formatted_glossary, csv_data_list = create_glossary(schema_data_with_metadata)

                    # Persist results into session_state immediately
                    st.session_state["formatted_glossary"] = formatted_glossary
                    st.session_state["business_glossary_df"] = generate_df(csv_data_list)

                # Force an immediate rerun so that Streamlit sees glossary_exists=True below
                st.rerun()

if "business_glossary_df" in st.session_state:
    st.subheader("Business Glossary", divider="rainbow")

    # (1) Show the stored markdown
    st.markdown(st.session_state["formatted_glossary"])

    # (2) Show Download-CSV
    business_glossary_df = st.session_state["business_glossary_df"]
    csv_content = business_glossary_df.to_csv(index=False)
    st.download_button(
        label="Download CSV",
        data=csv_content,
        file_name="Business Glossary.csv",
        mime="text/csv",
        key="download_glossary"
    )

    # (3) If this is Databricks, show “Add Comments to Unity Catalog”
    db_type = st.session_state["connections"][st.session_state["selected_connection"]]["type"]
    if db_type == "Databricks":
        if st.button("Add Comments to Unity Catalog", key="add_comments"):
            with st.spinner("Applying comments to Unity Catalog…"):
                success_count, failures = update_unity(
                    df=business_glossary_df,
                    selected_connection=st.session_state["selected_connection"],
                    session_state=st.session_state
                )

            st.success(f"Successfully added {success_count} column comments to Unity Catalog.")
            if failures:
                st.error(f"Failed to update {len(failures)} columns.")
                with st.expander("Show Errors"):
                    for tbl, col, err in failures:
                        st.text(f"{tbl}.{col} → {err}")


