import asyncio
import json
import os
import shutil
from typing import Optional

import pandas as pd
from langchain_core.tools import tool

from use_cases.discover import (
    apply_comments_to_unity_catalog,
    create_db_engine,
    create_glossary,
    create_purview_glossary,
    create_table_overview,
    fetch_purview_metadata,
    fetch_table_metadata,
    generate_csv,
    generate_df,
    get_catalogs_databricks,
    get_schemas,
    get_schemas_databricks,
    get_tables,
    get_tables_databricks,
    list_existing_purview_glossaries,
    list_purview_sql_tables,
    push_glossary_terms_to_purview,
    sanitize_term_name,
)

DISCOVER_AI_OUTPUT_DIR = "C:/EY/ey-genai-datanext-frontend/public"
os.makedirs(DISCOVER_AI_OUTPUT_DIR, exist_ok=True)


@tool
def discover_database_and_generate_glossary(
    db_type: str,
    host: Optional[str] = None,
    username: Optional[str] = None,
    password: Optional[str] = None,
    database: Optional[str] = None,
    port: Optional[str] = None,
    service: Optional[str] = None,
    authentication: Optional[str] = None,
) -> str:
    """
    Connects to a live database (PostgreSQL, MySQL, Microsoft SQL Server, OracleDB, SQLite),
    discovers all its non-system schemas and tables, and generates a comprehensive business glossary.
    Requires connection details appropriate for the specified db_type.
    Returns a summary message with the path to the generated CSV glossary.
    """
    print(
        f"--- TOOL: discover_database_and_generate_glossary called for db_type: {db_type} ---"
    )
    config = {k: v for k, v in locals().items() if v is not None}
    config["rdbms"] = config.pop("db_type")

    engine, error = create_db_engine(config)
    if error:
        return f"Connection Failed: {error}"

    try:
        all_schemas, error = get_schemas(engine)
        if error:
            return f"Schema Discovery Failed: {error}"
        print(f"Discovered schemas: {all_schemas}")

        # In a real tool, we might let the user select schemas/tables.
        # Here, we simplify by processing all discovered non-system schemas.
        # We need to construct the input for `create_table_overview`
        selected_schemas_per_connection = {"live_connection": all_schemas}
        selected_tables_per_schema = {}
        for schema in all_schemas:
            tables, tbl_error = get_tables(engine, schema)
            if tbl_error:
                continue
            selected_tables_per_schema[schema] = tables

        overview = create_table_overview(
            {"live_connection": {"engine": engine}},
            selected_schemas_per_connection,
            selected_tables_per_schema,
        )

        if not overview or not overview.get("live_connection"):
            return "No tables found or metadata could not be retrieved from the discovered schemas."

        # Generate glossary from the discovered metadata
        schema_data_json = json.dumps(overview)
        formatted_glossary, csv_data = create_glossary(schema_data_json)
        csv_content = generate_csv(csv_data)

        # Save CSV output
        db_name_for_file = database or service or "discovered_db"
        output_csv_filename = f"discovered_glossary_{db_name_for_file}.csv"
        output_csv_path = os.path.join(DISCOVER_AI_OUTPUT_DIR, output_csv_filename)
        with open(output_csv_path, "w", newline="") as f:
            f.write(csv_content)

        return f"Database discovery and glossary generation complete. CSV output saved to: {output_csv_path}"
    except Exception as e:
        return f"An error occurred during discovery and glossary generation: {e}"


# --- NEW DATABRICKS TOOL 1: FOR DISCOVERY ---
@tool
def discover_databricks_structure(
    server_hostname: str, http_path: str, access_token: str
) -> str:
    """
    Connects to a Databricks workspace and lists all available catalogs and their corresponding schemas.
    This tool is for exploration and does NOT generate a glossary.
    The output is a structured string for the user to review.
    """
    print("--- TOOL: discover_databricks_structure called ---")
    config = {
        "rdbms": "Databricks",
        "server_hostname": server_hostname,
        "http_path": http_path,
        "access_token": access_token,
    }

    try:
        engine, error = create_db_engine(config)
        if error:
            return f"Databricks Connection Failed: {error}"

        catalogs, cat_err = get_catalogs_databricks(engine)
        if cat_err:
            return f"Databricks Catalog Discovery Failed: {cat_err}"

        if not catalogs:
            return "No catalogs found in the Databricks workspace."

        # Build a human-readable string of the structure
        structure_report = [
            "Here are the catalogs and schemas found in your Databricks workspace:"
        ]
        for catalog in catalogs:
            # Skip system catalogs unless necessary
            if catalog.lower() in ["system", "information_schema"]:
                continue
            structure_report.append(f"\nCatalog: `{catalog}`")

            schemas, sch_err = get_schemas_databricks(engine, catalog)
            if sch_err:
                structure_report.append("  - (Could not fetch schemas)")
                continue

            if not schemas:
                structure_report.append("  - (No schemas found in this catalog)")

            for schema in schemas:
                structure_report.append(f"  - Schema: `{schema}`")

        structure_report.append(
            "\nPlease review the list and tell me which catalog and which specific schema(s) you want me to generate a glossary for."
        )

        return "\n".join(structure_report)

    except Exception as e:
        return f"An unexpected error occurred during Databricks discovery: {e}"


# --- NEW DATABRICKS TOOL 2: FOR GLOSSARY GENERATION ---
@tool
def generate_databricks_glossary_for_schemas(
    server_hostname: str, http_path: str, access_token: str, catalog: str, schemas: str
) -> str:
    """
    Generates a business glossary for all tables within one or more specific schemas in a given Databricks catalog.
    'schemas' should be a comma-separated string of schema names (e.g., 'default,sales_data').
    """
    print(
        f"--- TOOL: generate_databricks_glossary_for_schemas called for catalog '{catalog}' and schemas '{schemas}' ---"
    )
    try:
        # Robust path logic
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # output_base_dir = os.path.join(script_dir, '..', 'output', 'business glossary', 'discovered glossary')
        # os.makedirs(output_base_dir, exist_ok=True)

        config = {
            "rdbms": "Databricks",
            "server_hostname": server_hostname,
            "http_path": http_path,
            "access_token": access_token,
            "catalog": catalog,
        }

        schema_list = [s.strip() for s in schemas.split(",")]
        if not schema_list or all(s == "" for s in schema_list):
            return "Error: You must provide at least one valid schema name."

        engine, error = create_db_engine(config)
        if error:
            return f"Databricks Connection Failed: {error}"

        overview = {"live_connection": []}
        for schema in schema_list:
            tables, tbl_err = get_tables_databricks(engine, catalog, schema)
            if tbl_err:
                continue
            for table in tables:
                metadata, meta_err = fetch_table_metadata(
                    engine, table, schema, catalog, db_type="Databricks"
                )
                if meta_err:
                    continue
                formatted_meta = [
                    f"{col['name']}: ({col['type']}) [{col.get('constraints', '')}]"
                    for col in metadata.get("columns", [])
                ]
                overview["live_connection"].append(
                    {"Schema": schema, "Table": table, "Metadata": formatted_meta}
                )

        if not overview["live_connection"]:
            return f"No tables or metadata could be found in the specified schemas: {schemas}."

        schema_data_json = json.dumps(overview)
        _, csv_data = create_glossary(schema_data_json)
        if not csv_data:
            return "Error: The AI failed to generate a glossary in the correct format."

        df_glossary = generate_df(csv_data)

        # Save the final, user-facing CSV
        output_csv_filename = (
            f"discovered_glossary_databricks_{catalog}_{'_'.join(schema_list)}.csv"
        )
        output_csv_path = os.path.join(DISCOVER_AI_OUTPUT_DIR, output_csv_filename)
        df_glossary.to_csv(output_csv_path, index=False, encoding="utf-8")

        # --- NEW LOGIC: Save a temporary file for the next tool ---
        temp_glossary_path = os.path.join(
            DISCOVER_AI_OUTPUT_DIR, "temp_glossary_for_update.csv"
        )
        df_glossary.to_csv(temp_glossary_path, index=False, encoding="utf-8")

        # --- NEW LOGIC: Return a message that includes the temp path for the agent ---
        return (
            f"Glossary generation complete. The final file is saved at: {output_csv_path}. "
            f"The temporary file for Unity Catalog update is ready at: {temp_glossary_path}"
        )

    except Exception as e:
        import traceback

        traceback.print_exc()
        return f"An unexpected fatal error occurred during glossary generation: {e}"


# Update Unity catalog
@tool
def update_unity_catalog(
    server_hostname: str,
    http_path: str,
    access_token: str,
    catalog: str,
    schemas: str,
    glossary_csv_path: str,
) -> str:
    """
    Applies comments from a generated glossary CSV file to the specified tables and columns in Databricks Unity Catalog.
    """
    print(f"--- TOOL: update_unity_catalog called for catalog '{catalog}' ---")
    if not os.path.exists(glossary_csv_path):
        return f"Error: The glossary file was not found at the specified path: {glossary_csv_path}"

    try:
        config = {
            "rdbms": "Databricks",
            "server_hostname": server_hostname,
            "http_path": http_path,
            "access_token": access_token,
        }
        engine, error = create_db_engine(config)
        if error:
            return f"Failed to connect to Databricks: {error}"

        # Load the glossary data
        df_glossary = pd.read_csv(glossary_csv_path)
        schema_list = [s.strip() for s in schemas.split(",")]

        # Call the backend function to apply comments
        success_count, failures = apply_comments_to_unity_catalog(
            engine, catalog, schema_list, df_glossary
        )

        if failures:
            failed_str = "\n".join(
                [f"- {tbl}.{col}: {err}" for tbl, col, err in failures]
            )
            return (
                f"Partially completed. Successfully updated {success_count} comments. "
                f"Failed to update {len(failures)} comments. Failures:\n{failed_str}"
            )

        return f"Successfully applied {success_count} comments to Unity Catalog in catalog '{catalog}'."

    except Exception as e:
        import traceback

        traceback.print_exc()
        return f"An unexpected error occurred while updating Unity Catalog: {e}"


# --- PURVIEW TOOL 1: DISCOVER TABLES ---
@tool
def discover_purview_tables(
    tenant_id: str, client_id: str, client_secret: str, purview_name: str
) -> str:
    """Connects to Azure Purview and lists all available, registered SQL tables."""
    print(f"--- TOOL: discover_purview_tables for account: {purview_name} ---")
    config = {
        "rdbms": "Azure Purview",
        "tenant_id": tenant_id,
        "client_id": client_id,
        "client_secret": client_secret,
        "purview_name": purview_name,
    }
    try:
        client, error = create_db_engine(config)
        if error:
            return f"Purview Connection Failed: {error}"

        tables, err = list_purview_sql_tables(client)
        if err:
            return f"Error fetching tables: {err}"
        if not tables:
            return "No registered SQL tables were found in this Purview account."

        table_list = [f"- `{t['name']}`" for t in tables]
        return (
            "The following SQL tables were found in your Purview account:\n"
            + "\n".join(table_list)
            + "\n\nPlease specify which tables you want to generate a glossary for (you can say 'all' or list them, comma-separated)."
        )
    except Exception as e:
        return f"An unexpected error occurred: {e}"


# --- PURVIEW TOOL 2: GENERATE GLOSSARY ---
@tool
def generate_purview_glossary_for_tables(
    tenant_id: str,
    client_id: str,
    client_secret: str,
    purview_name: str,
    tables_to_process: str,
) -> str:
    """Generates a business glossary for specific tables discovered in Azure Purview."""
    print(
        f"--- TOOL: generate_purview_glossary_for_tables for tables: {tables_to_process} ---"
    )
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_base_dir = os.path.join(
            script_dir, "..", "output", "business glossary", "discovered glossary"
        )
        os.makedirs(output_base_dir, exist_ok=True)

        config = {
            "rdbms": "Azure Purview",
            "tenant_id": tenant_id,
            "client_id": client_id,
            "client_secret": client_secret,
            "purview_name": purview_name,
        }
        client, error = create_db_engine(config)
        if error:
            return f"Purview Connection Failed: {error}"

        all_tables, err = list_purview_sql_tables(client)
        if err:
            return f"Could not list Purview tables: {err}"

        selected_tables = []
        if tables_to_process.strip().lower() == "all":
            selected_tables = all_tables
        else:
            user_table_names = {t.strip().lower() for t in tables_to_process.split(",")}
            selected_tables = [
                tbl for tbl in all_tables if tbl["name"].lower() in user_table_names
            ]

        if not selected_tables:
            return f"Error: None of the specified tables ({tables_to_process}) could be found in Purview."

        overview = {"live_connection": []}
        for tbl in selected_tables:
            metadata, meta_err = fetch_purview_metadata(
                client, tbl["guid"], tbl["name"]
            )
            if meta_err:
                continue
            formatted_meta = [f"{col['name']}: ({col['type']})" for col in metadata]
            overview["live_connection"].append(
                {"Schema": "purview", "Table": tbl["name"], "Metadata": formatted_meta}
            )

        if not overview["live_connection"]:
            return "Could not fetch metadata for any of the selected tables."

        schema_data_json = json.dumps(overview)
        _, csv_data = create_glossary(schema_data_json)
        if not csv_data:
            return "Error: The AI failed to generate a glossary in the correct format."

        df_glossary = generate_df(csv_data)

        temp_glossary_path = os.path.join(
            DISCOVER_AI_OUTPUT_DIR, "temp_purview_glossary_for_update.csv"
        )
        df_glossary.to_csv(temp_glossary_path, index=False, encoding="utf-8")

        return (
            f"Glossary generation for {len(selected_tables)} Purview table(s) complete. "
            f"The temporary file for the Purview update is ready at: {temp_glossary_path}"
        )
    except Exception as e:
        return f"An unexpected error occurred during Purview glossary generation: {e}"


# --- PURVIEW TOOL 3: LIST GLOSSARIES ---
@tool
def list_purview_glossaries(
    tenant_id: str, client_id: str, client_secret: str, purview_name: str
) -> str:
    """Connects to Azure Purview and lists all existing glossaries."""
    print(f"--- TOOL: list_purview_glossaries for account: {purview_name} ---")
    config = {
        "rdbms": "Azure Purview",
        "tenant_id": tenant_id,
        "client_id": client_id,
        "client_secret": client_secret,
        "purview_name": purview_name,
    }
    client, error = create_db_engine(config)
    if error:
        return f"Purview Connection Failed: {error}"

    glossaries, err = list_existing_purview_glossaries(client)
    if err:
        return f"Error fetching glossaries: {err}"
    if not glossaries:
        return "No glossaries found in this Purview account. Please provide a name for a new glossary to be created."

    glossary_list = [f"- `{g['name']}`" for g in glossaries]
    return (
        "The glossary has been generated. Would you like me to push the terms to Purview? "
        "Please choose an existing glossary from the list below, or provide a new name to create one.\n\n"
        + "\n".join(glossary_list)
    )


# --- PURVIEW TOOL 4: PUSH TO GLOSSARY ---
@tool
def push_to_purview_glossary(
    tenant_id: str,
    client_id: str,
    client_secret: str,
    purview_name: str,
    glossary_name: str,
    glossary_csv_path: str,
) -> str:
    """Pushes terms from a CSV file to a specified Azure Purview glossary, creating it if it doesn't exist."""
    print(f"--- TOOL: push_to_purview_glossary for glossary: {glossary_name} ---")
    if not os.path.exists(glossary_csv_path):
        return f"Error: The glossary file was not found at {glossary_csv_path}. The agent must find this path from the conversation history."

    config = {
        "rdbms": "Azure Purview",
        "tenant_id": tenant_id,
        "client_id": client_id,
        "client_secret": client_secret,
        "purview_name": purview_name,
    }

    try:
        client, error = create_db_engine(config)
        if error:
            return f"Purview Connection Failed: {error}"

        # 1. Find or Create the Glossary
        glossaries, err = list_existing_purview_glossaries(client)
        if err:
            return f"Could not list existing glossaries: {err}"

        target_glossary_guid = None
        for g in glossaries:
            if g["name"].lower() == glossary_name.lower():
                target_glossary_guid = g["guid"]
                print(
                    f"Found existing glossary '{glossary_name}' with GUID: {target_glossary_guid}"
                )
                break

        if not target_glossary_guid:
            print(f"Glossary '{glossary_name}' not found. Creating it...")
            # The create_purview_glossary function from discover.py is what we should use here
            guid, create_err = create_purview_glossary(
                client,
                glossary_name,
                f"Glossary for {purview_name} assets",
                f"Auto-generated by DataNext Agent for {purview_name}.",
            )
            if create_err:
                return f"Failed to create new glossary '{glossary_name}': {create_err}"
            target_glossary_guid = guid
            print(
                f"Successfully created new glossary '{glossary_name}' with GUID: {target_glossary_guid}"
            )

        # 2. Read the PRE-GENERATED glossary file
        df_glossary = pd.read_csv(glossary_csv_path)

        # 3. Get a fresh list of all tables to ensure GUIDs are correct
        all_tables, list_err = list_purview_sql_tables(client)
        if list_err:
            return f"Could not get a fresh list of tables from Purview: {list_err}"

        # 4. Call the push function
        print(
            f"Pushing {len(df_glossary)} terms to glossary GUID {target_glossary_guid}..."
        )
        success_count, failures = push_glossary_terms_to_purview(
            client, target_glossary_guid, df_glossary, all_tables
        )

        if failures:
            failed_str = "\n".join(
                [f"- {tbl}.{col}: {err}" for tbl, col, err in failures]
            )
            return (
                f"Partially completed. Successfully pushed {success_count} terms. "
                f"Failed to push {len(failures)} terms. Failures:\n{failed_str}"
            )

        return f"Successfully pushed {success_count} terms to the '{glossary_name}' glossary in Azure Purview."

    except Exception as e:
        import traceback

        traceback.print_exc()
        return f"An unexpected error occurred while pushing to Purview: {e}"
