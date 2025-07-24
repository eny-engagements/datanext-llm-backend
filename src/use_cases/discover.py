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
from azure.identity import ClientSecretCredential
from azure.purview.catalog import PurviewCatalogClient
from databricks import sql as databricks_sql

def create_db_engine(config):
    drivers = {
        "Microsoft SQL Server": "mssql+pyodbc",
        "MySQL": "mysql+pymysql",
        "OracleDB": "oracle+cx_oracle",
        "SQLite": "sqlite",
        "PostgreSQL": "postgresql",
        "Databricks" : "databricks",
        "Azure Purview": "purview"
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
        elif config['rdbms'] == "Azure Purview":
            try:
                credential = ClientSecretCredential(
                    tenant_id=config['tenant_id'],
                    client_id=config['client_id'],
                    client_secret=config['client_secret']
                )
                client = PurviewCatalogClient(
                    endpoint=f"https://{config['purview_name']}.purview.azure.com",
                    credential=credential
                )
                return client, None
            except Exception as e:
                return None, f"Failed to connect to Azure Purview: {e}"
            
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

# In discover.py, add this new function

def apply_comments_to_unity_catalog(engine, catalog: str, schema_list: list, df: pd.DataFrame) -> tuple[int, list]:
    """
    Applies table-level and column-level comments to Unity Catalog.
    This is a standalone version designed to be called from a script or agent.

    Args:
        engine: The Databricks SQL connection engine.
        catalog (str): The name of the catalog to update.
        schema_list (list): A list of schemas that the comments belong to.
        df (pd.DataFrame): The DataFrame containing the glossary.
                           Must have columns: 'Table', 'Column', 'Definition', 'Table Description'.

    Returns:
        A tuple containing (success_count, failed_comments_list).
    """
    try:
        success_count = 0
        failed_comments = []
        updated_tables = set()  # track tables already updated with table-level description

        cursor = engine.cursor()

        for _, row in df.iterrows():

            table_name_raw = row['Table']
            table_schema = None
            clean_table_name = None

            for s in schema_list:
                if table_name_raw.lower().startswith(f"{s.lower()}."):
                    table_schema = s
                    clean_table_name = table_name_raw.split('.', 1)[1]
                    break
            
            if not table_schema:
                table_schema = schema_list[0]
                clean_table_name = table_name_raw

            column_name = row['Column']
            comment_text = str(row['Definition']).replace("'", "''")  # Escape single quotes for SQL
            table_description = str(row['Table Description']).replace("'", "''")  # Escape single quotes

            fq_table = f"`{catalog}`.`{table_schema}`.`{clean_table_name}`"

            # Step 1: Set table-level comment (only once per table)
            if fq_table not in updated_tables and pd.notna(row['Table Description']):
                try:
                    table_comment_query = f"""COMMENT ON TABLE {fq_table} IS '{table_description}'"""
                    cursor.execute(table_comment_query)
                    updated_tables.add(fq_table)
                except Exception as e:
                    failed_comments.append((fq_table, "[TABLE_DESC]", str(e)))

            # Step 2: Set column-level comment
            try:
                col_comment_query = f"""COMMENT ON COLUMN {fq_table}.`{column_name}` IS '{comment_text}'"""
                cursor.execute(col_comment_query)
                success_count += 1
            except Exception as e:
                failed_comments.append((fq_table, column_name, str(e)))

        cursor.close()
        return success_count, failed_comments

    except Exception as e:
        import traceback
        traceback.print_exc()
        return 0, [("GENERAL_ERROR", "N/A", str(e))]

def list_purview_sql_tables(client):
    try:
        search_request = {
            "keywords": "*",
            "filter": {"entityType": "azure_sql_table"},
            "limit": 50
        }
        response = client.discovery.query(search_request=search_request)
        tables = response.get("value", [])

        results = []
        for tbl in tables:
            results.append({
                "name": tbl.get("name"),
                "guid": tbl.get("id")
            })
        return results, None
    except Exception as e:
        return None, f"Error fetching tables from Purview: {e}"

# In discover.py, add these two new functions

def apply_terms_to_purview_glossary(client, glossary_guid: str, df: pd.DataFrame, all_tables_in_purview: list) -> tuple[int, list]:
    """
    Creates and assigns glossary terms in Azure Purview from a DataFrame.

    Args:
        client: The authenticated PurviewCatalogClient.
        glossary_guid (str): The GUID of the target Purview glossary.
        df (pd.DataFrame): DataFrame with columns 'Table', 'Column', 'Definition'.
        all_tables_in_purview (list): A list of dicts [{'name': '...', 'guid': '...'}] from list_purview_sql_tables.

    Returns:
        A tuple containing (success_count, failed_terms_list).
    """
    def normalize(name: str) -> str:
        # Helper to normalize table names for matching
        return name.strip().lower().lstrip('.').replace('"', '')

    success = 0
    failed = []

    # Create a mapping from normalized table name to GUID for fast lookups
    table_name_to_guid = {
        normalize(table['name']): table['guid'] for table in all_tables_in_purview
    }

    for _, row in df.iterrows():
        table_name_raw = row["Table"]
        column_name_raw = row["Column"]
        definition = row["Definition"]

        # Find the table's GUID
        normalized_table = normalize(table_name_raw)
        table_guid = table_name_to_guid.get(normalized_table)
        
        if not table_guid:
            failed.append((table_name_raw, column_name_raw, "❌ Table not found in Purview or no GUID available"))
            continue

        # Sanitize term names to be valid in Purview
        base_term_name = sanitize_term_name(column_name_raw)
        table_prefix = sanitize_term_name(table_name_raw)
        attempt = 0
        max_attempts = 2 # Try with and without a table prefix

        while attempt < max_attempts:
            new_term_name = base_term_name if attempt == 0 else f"{table_prefix}_{base_term_name}"
            payload = {
                "name": new_term_name,
                "shortDescription": definition[:250], # Max length for short desc
                "longDescription": definition,
                "anchor": {"glossaryGuid": glossary_guid},
                "status": "Approved"
            }

            try:
                # 1. Create the glossary term
                term = client.glossary.create_glossary_term(payload)
                term_guid = term.get("guid")
                if term_guid:
                    # 2. Assign the term to the table entity
                    client.glossary.assign_term_to_entities(term_guid, [{
                        "guid": table_guid,
                        "typeName": "azure_sql_table" # Or other relevant types
                    }])
                    success += 1
                    break # Success, exit the while loop
                else:
                    failed.append((table_name_raw, new_term_name, "❌ Term created, but no GUID returned"))
                    break # Exit while loop

            except Exception as e:
                error_msg = str(e)
                if "already exists" in error_msg and attempt == 0:
                    attempt += 1  # Retry with prefixed name
                elif "name cannot contain" in error_msg:
                    failed.append((table_name_raw, new_term_name, f"❌ Invalid characters in term name: {new_term_name}"))
                    break
                else:
                    failed.append((table_name_raw, new_term_name, f"❌ Exception: {error_msg}"))
                    break
    
    return success, failed


def list_existing_purview_glossaries(client) -> tuple[list, str | None]:
    """Lists existing glossaries in a Purview account."""
    try:
        glossaries = client.glossary.list_glossaries()
        # Return a list of dicts with name and guid
        return [{"name": g["name"], "guid": g["guid"]} for g in glossaries], None
    except Exception as e:
        return [], f"Error listing glossaries: {e}"

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
        if hasattr(engine, "cursor"):  # Databricks check
            if catalog:
                return get_schemas_databricks(engine, catalog)
            else:
                return None, "Missing catalog for Databricks connection."
            
        elif engine.name == 'sqlite':
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

def fetch_purview_metadata(client, table_guid, table_name):
    try:
        tbl_full = client.entity.get_by_guid(guid=table_guid)
        referred = tbl_full.get("referredEntities", {})

        columns = []
        for col_guid, col_ent in referred.items():
            if col_ent.get("typeName") == "azure_sql_column":
                attrs = col_ent.get("attributes", {})
                columns.append({
                    "name": attrs.get("name"),
                    "type": attrs.get("data_type"),
                    "nullable": attrs.get("is_nullable", True),
                    "description": attrs.get("description", "")
                })
        return columns, None
    except Exception as e:
        return None, f"Failed to fetch metadata for {table_name}: {e}"


def get_tables(engine, schema=None, catalog=None, db_type=None):
    try:
        if hasattr(engine, "cursor"):  # Databricks
            if not catalog:
                return None, "Missing catalog for Databricks table fetch."
            return get_tables_databricks(engine, catalog, schema)

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
    from streamlit.runtime.scriptrunner import get_script_run_ctx
    import streamlit as st

    overview = {connection_key: [] for connection_key in connections.keys()}

    for connection_key, connection_info in connections.items():
        engine = connection_info['engine']
        db_type = db_type_lookup.get(connection_key, "")

        if db_type == "Azure Purview":
            selected_guids = st.session_state.get('purview_selected_guids', [])
            selected_names = st.session_state.get('purview_selected_names', [])

            for table_name, table_guid in zip(selected_names, selected_guids):
                metadata, error = fetch_purview_metadata(engine, table_guid, table_name)
                if error:
                    continue

                formatted_metadata = []
                for col in metadata:
                    column_info = f"{col['name']}: ({col['type']})"
                    constraints = []
                    if col.get("nullable") is False:
                        constraints.append("NOT NULL")
                    if col.get("description"):
                        constraints.append(col["description"])
                    if constraints:
                        column_info += f" [{' | '.join(constraints)}]"
                    formatted_metadata.append(column_info)

                table_info = {
                    "Schema": "",  # No schemas in Purview
                    "Table": table_name,
                    "Metadata": formatted_metadata
                }
                overview[connection_key].append(table_info)

        else:
            for schema in selected_schemas_per_connection.get(connection_key, []):
                selected_tables = selected_tables_per_schema_per_connection.get(schema, [])
                catalog = catalog_lookup.get(connection_key)
                db_type = db_type_lookup.get(connection_key)

                for table in selected_tables:
                    metadata, error = fetch_table_metadata(engine, table, schema=schema ,catalog=catalog, db_type=db_type)
                    if error:
                        continue

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
                    overview[connection_key].append(table_info)

    return overview

def fetch_table_guids_from_purview(client, account_name, schema, tables):
    results = client.discovery.get_tables(account_name=account_name, schema=schema)
    table_guids = []
    table_names = []
    for table in results:
        if table["name"] in tables:
            table_names.append(table["name"])
            table_guids.append(table["id"])  # or table["guid"] depending on your SDK
    return table_names, table_guids


# In discover.py

def chunk_schema(schema_data):
    """
    Chunks the schema data for processing, but intelligently handles placeholder schemas.
    """
    schema_batches = []

    for tablespace, table_list in schema_data.items():
        # table_list is a list of dicts like { "Schema": "...", "Table": "...", "Metadata": […] }

        single_connection_chunks = []
        for item in table_list:
            # Always start a brand-new chunk with the “Tablespace:” line
            chunk = f"Tablespace: {tablespace}\n"

            # --- THIS IS THE FIX ---
            # If the schema is a placeholder like "purview", do not prepend it to the table name.
            # Otherwise, for real databases, keep the schema prefix.
            if item['Schema'] and item['Schema'].lower() not in ['purview', 'main']:
                 chunk += f"Table: {item['Schema']}.{item['Table']}\n"
            else:
                 # For Purview and SQLite, just use the table name directly.
                 chunk += f"Table: {item['Table']}\n"
            # --- END FIX ---

            chunk += f"Metadata: {item['Metadata']}\n"
            single_connection_chunks.append(chunk)

        schema_batches.append(single_connection_chunks)

    return schema_batches


import pandas as pd
import traceback

def update_unity(engine, catalog: str, schema_list: list, df: pd.DataFrame) -> tuple[int, list]:
    """
    Applies table-level and column-level comments to Unity Catalog.
    This is a helper function designed to be called by an agent tool.
    
    Args:
        engine: The Databricks SQL connection object.
        catalog (str): The name of the catalog to update.
        schema_list (list): A list of schemas that the comments belong to.
        df (pd.DataFrame): The DataFrame containing the glossary.
                           Must have columns: 'Table', 'Column', 'Definition', 'Table Description'.

    Returns:
        A tuple containing (success_count, failed_comments_list).
    """
    success_count = 0
    failed_comments = []
    updated_tables = set()

    if 'Table Description' not in df.columns:
        df['Table Description'] = None

    cursor = None # Initialize cursor to None
    try:
        # --- FIX IS HERE ---
        # The 'engine' object is already the connection. Get the cursor directly from it.
        cursor = engine.cursor()

        for _, row in df.iterrows():
            table_name_raw = row['Table']
            column_name = row['Column']
            print("tablename" , table_name_raw)
            print("columnname", column_name)
            comment_text = str(row['Definition']).replace("'", "''") 
            table_description = str(row.get('Table Description', '')).replace("'", "''")

            table_schema = None
            clean_table_name = None

            for s in schema_list:
                if str(table_name_raw).lower().startswith(f"{s.lower()}."):
                    table_schema = s
                    clean_table_name = table_name_raw.split('.', 1)[1]
                    break
            
            if not table_schema:
                table_schema = schema_list[0]
                clean_table_name = table_name_raw

            fq_table = f"{catalog}.{clean_table_name}"
            print("fq_table", fq_table)
            # Set table-level comment
            if fq_table not in updated_tables and table_description and pd.notna(row.get('Table Description')):
                try:
                    table_comment_query = f"""COMMENT ON TABLE {fq_table} IS '{table_description}'"""
                    cursor.execute(table_comment_query)
                    updated_tables.add(fq_table)
                except Exception as e:
                    failed_comments.append((fq_table, "[TABLE_DESC]", str(e)))

            # Set column-level comment
            if pd.notna(column_name) and comment_text:
                try:
                    col_comment_query = f"""COMMENT ON COLUMN {fq_table}.`{column_name}` IS '{comment_text}'"""
                    cursor.execute(col_comment_query)
                    success_count += 1
                except Exception as e:
                    failed_comments.append((fq_table, column_name, str(e)))

        return success_count, failed_comments

    except Exception as e:
        traceback.print_exc()
        return 0, [("GENERAL_ERROR", "N/A", str(e))]
    finally:
        # --- IMPORTANT ---
        # Ensure resources are always closed to prevent leaks
        if cursor:
            cursor.close()
        if engine:
            engine.close()
# def update_unity(df, selected_connection, session_state):
#     """
#     Applies table-level and column-level comments to Unity Catalog using the Databricks SQL connector.
#     """
#     try:
#         engine = session_state['connections'][selected_connection]['engine']
#         catalog = session_state['selected_catalog']
#         schema_list = session_state['selected_schemas_per_connection'][selected_connection]

#         success_count = 0
#         failed_comments = []
#         updated_tables = set()  # track tables already updated with table-level description

#         cursor = engine.cursor()

#         for _, row in df.iterrows():
#             schema = None

#             # Detect schema from fully qualified table name
#             for possible_schema in schema_list:
#                 if row['Table'].startswith(possible_schema + "."):
#                     schema = possible_schema
#                     # print("schema", schema)
#                     table_name = row['Table'].split(".")[1]
#                     break

#             if not schema:
#                 schema = schema_list[0]  # fallback
#                 # print("schema ###", schema)
#                 table_name = row['Table']
#                 # print("table name" , table_name)

#             column_name = row['Column']
#             comment_text = row['Definition'].replace("'", "")  # escape single quotes
#             table_description = row['Table Description'].replace("'", "")  # escape quotes

#             fq_table = f"{catalog}.{table_name}"
#             # print("table path" , fq_table)

#             # Step 1: Set table-level comment (only once per table)
#             if fq_table not in updated_tables:
#                 try:
#                     table_comment_query = f"""COMMENT ON TABLE {fq_table} IS '{table_description}'"""
#                     # print(f"[TABLE] {table_comment_query}")
#                     cursor.execute(table_comment_query)
#                     updated_tables.add(fq_table)
#                 except Exception as e:
#                     failed_comments.append((table_name, "[TABLE_DESC]", str(e)))

#             # Step 2: Set column-level comment
#             try:
#                 col_comment_query = f"""COMMENT ON COLUMN {fq_table}.{column_name} IS '{comment_text}'"""
#                 # print(f"[COLUMN] {col_comment_query}")
#                 cursor.execute(col_comment_query)
#                 success_count += 1
#             except Exception as e:
#                 failed_comments.append((table_name, column_name, str(e)))

#         cursor.close()
#         return success_count, failed_comments

#     except Exception as e:
#         return 0, [("GENERAL_ERROR", "N/A", str(e))]

#     except Exception as e:
#         print('Error')
#         return 0, [("GENERAL_ERROR", "N/A", str(e))]


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



def list_purview_glossaries(client):
    try:
        glossaries = client.glossary.list_glossaries()
        return glossaries, None
    except Exception as e:
        return None, f"Error listing glossaries: {e}"

def create_purview_glossary(client, name, short_desc, long_desc):
    try:
        glossary = client.glossary.create_glossary({
            "name": name,
            "shortDescription": short_desc,
            "longDescription": long_desc
        })
        return glossary["guid"], None
    except Exception as e:
        return None, f"Error creating glossary: {e}"

import re

def sanitize_term_name(name: str) -> str:
    """Remove or replace disallowed characters for glossary term names."""
    name = name.strip()
    name = re.sub(r"[ @\.,\(\)]", "_", name)  
    name = re.sub(r"[^A-Za-z0-9_]", "", name)  
    name = re.sub(r"_+", "_", name) 
    return name.strip("_")

def normalize(name: str) -> str:
    return name.strip().lower().lstrip('.').replace('"', '')

# In discover.py, REPLACE the push_glossary_terms_to_purview function

# We rename the last argument to reflect what it actually is

# In discover.py, this is the CORRECT version of the function.
# Make sure this is the active implementation.

def push_glossary_terms_to_purview(client, glossary_guid, df, all_tables_in_purview: list):
    
    success = 0
    failed = []

    # --- THIS IS THE FIX ---
    # It correctly builds the dictionary from the list that the agent passes in.
    normalized_guid_map = {
        normalize(table['name']): table['guid'] for table in all_tables_in_purview
    }
    # -----------------------

    for _, row in df.iterrows():
        table_name_raw = row["Table"]
        column_name_raw = row["Column"]
        definition = row["Definition"]

        # The rest of the function can now work correctly
        normalized_table = normalize(table_name_raw)
        table_guid = normalized_guid_map.get(normalized_table)
        if not table_guid:
            failed.append((table_name_raw, column_name_raw, f"❌ Table '{table_name_raw}' not found in Purview's asset list."))
            continue

        base_term_name = sanitize_term_name(column_name_raw)
        table_prefix = sanitize_term_name(table_name_raw)
        attempt = 0
        max_attempts = 5

        while attempt < max_attempts:
            term_candidate = base_term_name if attempt == 0 else f"{table_prefix}_{base_term_name}"
            new_term_name = term_candidate[:255] # Truncate to be safe

            payload = {
                "name": new_term_name,
                "shortDescription": definition[:250],
                "longDescription": definition,
                "anchor": {"glossaryGuid": glossary_guid},
                "status": "Approved"
            }

            try:
                term = client.glossary.create_glossary_term(payload)
                term_guid = term.get("guid")
                if term_guid:
                    client.glossary.assign_term_to_entities(term_guid, [{
                        "guid": table_guid,
                        "typeName": "azure_sql_table"
                    }])
                    success += 1
                    break
                else:
                    failed.append((table_name_raw, new_term_name, "❌ Term created, but no GUID returned"))
                    break

            except Exception as e:
                error_msg = str(e)
                if "already exists" in error_msg.lower() and attempt < max_attempts - 1:
                    attempt += 1
                elif "name cannot contain" in error_msg:
                    failed.append((table_name_raw, new_term_name, "❌ Invalid characters in name after sanitization"))
                    break
                else:
                    failed.append((table_name_raw, new_term_name, f"❌ Exception: {error_msg}"))
                    break

    return success, failed





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
# Add to database type options
        db_type = st.selectbox("Database Type", ["PostgreSQL", "MySQL", "Microsoft SQL Server", "OracleDB", "SQLite", "Databricks", "Azure Purview"])
        
        connection_details = {}
        
        if db_type == "Databricks":
            connection_details['server_hostname'] = st.text_input("Server Hostname (e.g. adb-xxx.azuredatabricks.net)")
            connection_details['http_path'] = st.text_input("HTTP Path (from SQL warehouse)")
            connection_details['access_token'] = st.text_input("Access Token", type="password")

        elif db_type == "SQLite":
            connection_details['database'] = st.text_input("Database File Path")
                
                
        if db_type == "Azure Purview":
            connection_details['tenant_id'] = st.text_input("Tenant ID", value="", type="password")
            connection_details['client_id'] = st.text_input("Client ID", value="", type="password")
            connection_details['client_secret'] = st.text_input("Client Secret", value="", type="password")
            connection_details['purview_name'] = st.text_input("Purview Account Name", value="DataNext")

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
            

            elif db_type == "Azure Purview":
                connection_key = f"Azure Purview > {connection_details['purview_name']}"
                if engine:
                    st.session_state['connections'][connection_key] = {
                        'engine': engine,
                        'schemas': [],  # Will be fetched later
                        'type': db_type
                    }
                    st.session_state['selected_connection'] = connection_key
                    st.success("Azure Purview connection established.")
                else:
                    st.error(error)

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

            elif db_type == "Azure Purview":
                client = connection_info['engine']

                with st.spinner("Fetching registered SQL tables from Purview..."):
                    purview_tables, err = list_purview_sql_tables(client)

                if err:
                    st.error(err)
                elif not purview_tables:
                    st.info("No Azure SQL tables found in Purview.")
                else:
                    # build name ↔ guid map
                    names    = [t["name"] for t in purview_tables]
                    guid_map = {t["name"]: t["guid"] for t in purview_tables}

                    # load last selection (or empty list)
                    last = st.session_state.get("purview_selected_names", [])

                    # Persist multiple selections with stable state
                    selected = st.multiselect(
                        "Select Tables",
                        options=names,
                        default=last,
                        key="purview_tables"
                    )

                    # Only update GUIDs if selection changed and keys exist
                    if selected != st.session_state.get("purview_selected_names", []):
                        updated_guids = []
                        for name in selected:
                            if name in guid_map:
                                updated_guids.append(guid_map[name])
                            else:
                                st.warning(f"Missing GUID for table: {name}")
                        
                        # Persist only if all GUIDs resolved
                        if len(updated_guids) == len(selected):
                            st.session_state["purview_selected_names"] = selected
                            st.session_state["purview_selected_guids"] = updated_guids


            elif db_type == "SQLite":
                selected_schemas = ['main']
                st.info("SQLite does not support multiple schemas. Using 'main' schema.")
                st.session_state['selected_schemas_per_connection'][selected_connection] = selected_schemas
                for schema in selected_schemas:
                    tables, table_error = get_tables(engine, schema=schema)
                    if table_error:
                        st.error(table_error)
                    else:
                        selected_tables = st.session_state['selected_tables_per_schema_per_connection'].get(schema, [])
                        new_selected_tables = st.multiselect(
                            f"Choose tables for schema: {schema}", tables, default=selected_tables, key=schema
                        )
                        st.session_state['selected_tables_per_schema_per_connection'][schema] = new_selected_tables

            else:
                selected_schemas = st.multiselect("Select Schemas", schemas)
                st.session_state['selected_schemas_per_connection'][selected_connection] = selected_schemas
                for schema in selected_schemas:
                    tables, table_error = get_tables(engine, schema=schema)
                    if table_error:
                        st.error(table_error)
                    else:
                        selected_tables = st.session_state['selected_tables_per_schema_per_connection'].get(schema, [])
                        new_selected_tables = st.multiselect(
                            f"Choose tables for schema: {schema}", tables, default=selected_tables, key=schema
                        )
                        st.session_state['selected_tables_per_schema_per_connection'][schema] = new_selected_tables


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

                if db_type != "Azure Purview":
                    if st.button(f"Preview {item['Table']}", key=f"preview-{connection_key} > {item['Schema']} > {item['Table']}"):
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

    # (4) If this is Azure Purview, allow pushing to a glossary
    if db_type == "Azure Purview":
        client = st.session_state['connections'][st.session_state['selected_connection']]['engine']
        table_names = st.session_state.get("purview_selected_names", [])
        table_guids = st.session_state.get("purview_selected_guids", [])
        table_name_to_guid = {
            name.strip().lstrip('.').lower(): guid
            for name, guid in zip(table_names, table_guids)
        }
        glossaries, err = list_purview_glossaries(client)
        if err:
            st.error(err)
        else:
            existing_names = [g["name"] for g in glossaries]
            glossary_mode = st.radio("Select Glossary Action", ["Use Existing", "Create New"], horizontal=True)

            selected_guid = None
            if glossary_mode == "Use Existing":
                selected_glossary = st.selectbox("Choose Existing Glossary", existing_names)
                selected_guid = next(g["guid"] for g in glossaries if g["name"] == selected_glossary)
            else:
                new_name = st.text_input("Glossary Name")
                short_desc = st.text_area("Short Description")
                long_desc = st.text_area("Long Description")
                if st.button("Create Glossary"):
                    new_guid, err = create_purview_glossary(client, new_name, short_desc, long_desc)
                    if err:
                        st.error(err)
                    else:
                        selected_guid = new_guid
                        st.success(f"✅ Created glossary '{new_name}'")

            if selected_guid and st.button("Push to Purview Glossary"):
                with st.spinner("Pushing glossary terms..."):
                    success_count, failures = push_glossary_terms_to_purview(
                        client=client,
                        glossary_guid=selected_guid,
                        df=business_glossary_df,
                        table_name_to_guid=table_name_to_guid
                    )

                st.success(f"Pushed {success_count} terms to glossary.")
                if failures:
                    st.error(f"❌ Failed to push {len(failures)} terms.")
                    with st.expander("Show Errors"):
                        for tbl, term, err in failures:
                            st.text(f"{tbl}.{term} → {err}")