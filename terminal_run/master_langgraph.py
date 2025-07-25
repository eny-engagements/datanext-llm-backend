import asyncio
import json
import os
import shutil
from typing import Optional

import pandas as pd
from define import create_business_glossary_excel as actual_define_create_excel
from define import parallel_description as actual_define_parallel_description
from define import read_schema_in_chunks_and_batch as actual_define_read_schema
from discover import (
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
    update_unity,
)
from etl_documentation import analyse_sp as actual_lineage_extractor_analyse_sp
from etl_documentation import (
    create_etl_documentation_excel as actual_lineage_extractor_create_excel,
)
from etl_documentation import get_glossary as actual_lineage_extractor_get_glossary
from etl_documentation import visualise_etl as actual_lineage_extractor_visualise_etl
from langchain_core.tools import tool

DEFINE_AI_OUTPUT_DIR = "./output/business glossary"
LINEAGE_EXTRACTOR_OUTPUT_DIR = "./output/lineage"
DISCOVER_AI_OUTPUT_DIR = "./output/discovered glossary"
os.makedirs(DEFINE_AI_OUTPUT_DIR, exist_ok=True)
os.makedirs(LINEAGE_EXTRACTOR_OUTPUT_DIR, exist_ok=True)
os.makedirs(DISCOVER_AI_OUTPUT_DIR, exist_ok=True)


@tool
def define_data_model_from_excel(excel_path: str, schema_description: str) -> str:
    """
    Creates a business glossary from an input Excel file...
    """
    print(
        f"--- TOOL: define_data_model_from_excel called with excel_path: {excel_path} ---"
    )
    if not os.path.exists(excel_path):
        return f"Error: Input Excel file not found at '{excel_path}'."
    if not schema_description:
        return "Error: Schema description is required."

    try:
        # --- FIX: Ensure output directory exists ---
        os.makedirs(DEFINE_AI_OUTPUT_DIR, exist_ok=True)
        # --- END FIX ---

        _, schema_info_batches = actual_define_read_schema(excel_path)
        all_glossary_parts = []
        for batch in schema_info_batches:
            glossary_part = actual_define_parallel_description(
                schema_description, batch, data_dictionary=[]
            )
            all_glossary_parts.append(glossary_part)
        final_glossary_markdown = "  \n".join(all_glossary_parts)

        schema_name_from_path = os.path.basename(excel_path)
        # Note: Your actual_define_create_excel also saves the file. We'll rely on its path logic.
        actual_define_create_excel(schema_name_from_path, final_glossary_markdown)

        output_excel_filename = (
            schema_name_from_path.replace(".xlsx", "") + "_with_Business_Glossary.xlsx"
        )
        output_excel_path = os.path.join(DEFINE_AI_OUTPUT_DIR, output_excel_filename)

        return f"Business Glossary generation complete. Excel output saved to: {output_excel_path}"
    except Exception as e:
        return f"Error in define_data_model_from_excel: {e}"


@tool
def extract_etl_lineage_and_documentation(
    script_path: str,
    dialect: str,
    additional_context: Optional[str] = "N/A",
    business_glossary_excel_path: Optional[str] = None,
) -> str:
    """
    Analyzes an ETL script (SQL or text file) to generate documentation, a data lineage diagram,
    and an Excel report.
    Required inputs are the path to the script file and the SQL dialect (e.g., 'T-SQL', 'Spark SQL').
    Optional inputs include additional natural language context and the path to a business glossary Excel file.
    Returns a summary message with paths to the generated lineage diagram and Excel documentation.
    """
    print(
        f"--- TOOL: extract_etl_lineage_and_documentation called with script_path: {script_path}, dialect: {dialect} ---"
    )
    if not os.path.exists(script_path):
        return f"Error: Input script file not found at '{script_path}'."
    if not dialect:
        return "Error: SQL dialect is required (e.g., 'T-SQL', 'Spark SQL')."

    full_context = additional_context
    if business_glossary_excel_path:
        if not os.path.exists(business_glossary_excel_path):
            print(
                f"Warning: Glossary file '{business_glossary_excel_path}' not found, proceeding without it."
            )
        else:
            try:
                with open(script_path, "r", encoding="utf-8") as f:
                    script_content = f.read()
                glossary_context_str = actual_lineage_extractor_get_glossary(
                    script_content, business_glossary_excel_path
                )
                full_context += f"\n\nBusiness Glossary (from {os.path.basename(business_glossary_excel_path)}):\n{glossary_context_str}"
            except Exception as e:
                print(
                    f"Warning: Could not process glossary file '{business_glossary_excel_path}': {e}. Proceeding without it."
                )

    try:
        report_name = os.path.basename(script_path)
        with open(script_path, "r", encoding="utf-8") as f:
            script_content_for_analysis = f.read()

        df_analysis, df_filled_analysis = actual_lineage_extractor_analyse_sp(
            script_content_for_analysis, full_context, dialect
        )
        diagram_path = actual_lineage_extractor_visualise_etl(
            script_content_for_analysis, report_name, df_analysis
        )
        actual_lineage_extractor_create_excel(df_filled_analysis, report_name)
        output_excel_filename = report_name.replace(
            ".txt", " ETL Documentation.xlsx"
        ).replace(".sql", " ETL Documentation.xlsx")
        output_excel_path = os.path.join(
            LINEAGE_EXTRACTOR_OUTPUT_DIR, output_excel_filename
        )

        # The markdown table (df_analysis.to_markdown()) could be returned if the agent is prompted to display it.
        return (
            f"ETL Lineage and Documentation generation complete. "
            f"Lineage Diagram saved to: {diagram_path}. "
            f"Excel documentation saved to: {output_excel_path}."
        )
    except Exception as e:
        return f"Error in extract_etl_lineage_and_documentation: {e}"


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
        csv_content = generate_df(csv_data)
        # Save CSV output
        db_name_for_file = database or service or "discovered_db"
        output_csv_filename = f"discovered_glossary_{db_name_for_file}.csv"
        output_csv_path = os.path.join(DISCOVER_AI_OUTPUT_DIR, output_csv_filename)
        csv_content.to_csv(output_csv_path, index=False, encoding="utf-8")

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
        output_base_dir = os.path.join(
            script_dir, "..", "output", "business glossary", "discovered glossary"
        )
        os.makedirs(output_base_dir, exist_ok=True)

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
        output_csv_path = os.path.join(output_base_dir, output_csv_filename)
        df_glossary.to_csv(output_csv_path, index=False, encoding="utf-8")

        # --- NEW LOGIC: Save a temporary file for the next tool ---
        temp_glossary_path = os.path.join(
            output_base_dir, "temp_glossary_for_update.csv"
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
        success_count, failures = update_unity(
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
            output_base_dir, "temp_purview_glossary_for_update.csv"
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


# Step 2
from langchain_openai import AzureChatOpenAI
from langgraph.prebuilt import create_react_agent

agent_llm = AzureChatOpenAI(
    azure_deployment="gpt-4o",
    openai_api_version="2024-02-15-preview",
    api_key="901cce4a4b4045b39727bf1ce0e36219",
    azure_endpoint="https://aifordata-azureopenai.openai.azure.com/",
    temperature=0,
    max_tokens=4096,
)

define_agent = create_react_agent(
    model=agent_llm,
    tools=[define_data_model_from_excel],
    prompt=(
        "You are a service agent named DefineAgent. Your only job is to execute the `define_data_model_from_excel` tool. "
        "Use the `excel_path` and `schema_description` from the conversation history to call the tool. "
        "**After the tool runs, you MUST report the result back to the user as your final answer.**"
    ),
    name="DefineAgent",
)

lineage_extractor_agent = create_react_agent(
    model=agent_llm,
    tools=[extract_etl_lineage_and_documentation],
    prompt=(
        "You are the LineageExtractorAgent, an expert in analyzing ETL scripts (SQL or Spark SQL) "
        "to produce documentation and data lineage diagrams. You can optionally use a "
        "business glossary (Excel file) for better context. "
        "To use the 'extract_etl_lineage_and_documentation' tool, you *must* have the path to the script file "
        "and the SQL dialect. A business glossary path and additional context are optional. "
        "If the script path or dialect is missing, ask the user to provide them clearly. "
        "If they mention a glossary or context, try to use that too."
    ),
    name="LineageExtractorAgent",
)

discover_agent = create_react_agent(
    model=agent_llm,
    tools=[
        # RDBMS Tool
        discover_database_and_generate_glossary,
        # Databricks Tools
        discover_databricks_structure,
        generate_databricks_glossary_for_schemas,
        update_unity_catalog,
        # Purview Tools - ADD THESE
        discover_purview_tables,
        generate_purview_glossary_for_tables,
        list_purview_glossaries,
        push_to_purview_glossary,
    ],
    prompt=(
        "You are the DiscoverAgent, a specialist in generating glossaries and updating data catalogs.\n"
        "You must guide the user through a multi-step process depending on the data source.\n\n"
        "**Databricks Workflow:**\n"
        "1. Call `discover_databricks_structure` to list catalogs/schemas.\n"
        "2. Ask user to select catalog/schemas.\n"
        "3. Call `generate_databricks_glossary_for_schemas`.\n"
        "4. The tool returns a temp file path. Ask the user if they want to update Unity Catalog.\n"
        "5. If 'yes' or a input which means that you should push the glossary, call `update_unity_catalog` with all required arguments, including the temp file path from conversation history.\n\n"
        "**Azure Purview Workflow:**\n"
        "1. First, call `discover_purview_tables` to list all available tables.\n"
        "2. Present this list to the user and ask them to select the tables they want (or 'all').\n"
        "3. Next, call `generate_purview_glossary_for_tables` with the user's selection.\n"
        "4. This tool will return a temp file path. You MUST then call the `list_purview_glossaries` tool to get a list of existing glossaries.\n"
        "5. Present the list of glossaries to the user and ask them to choose one or provide a new name.\n"
        "6. Once the user provides a glossary name, you MUST call the `push_to_purview_glossary` tool with all required arguments, including the temp file path and the user's chosen glossary name.\n\n"
        "**RDBMS Workflow:**\n"
        "- Call `discover_rdbms_and_generate_glossary` and report the result. This is a single step.\n\n"
        "**IMPORTANT:** For multi-step workflows (Databricks, Purview), you must remember information like file paths from previous tool calls in the conversation to use as arguments for subsequent tool calls."
    ),
    name="DiscoverAgent",
)
from langchain_core.messages import AIMessage, HumanMessage
from langgraph_supervisor import create_supervisor

supervisor_llm = AzureChatOpenAI(
    model="gpt-4o",
    temperature=0,
    openai_api_version="2024-02-15-preview",
    api_key="901cce4a4b4045b39727bf1ce0e36219",
    azure_endpoint="https://aifordata-azureopenai.openai.azure.com/",
)

members = ["DefineAgent", "LineageExtractorAgent", "DiscoverAgent"]

supervisor_system_prompt = (
    "You are the Master AI Agent. Your role is to manage a conversation between a user "
    "and a team of specialized AI agents: 'DefineAgent', 'LineageExtractorAgent', and 'DiscoverAgent'.\n\n"
    "**Agent Capabilities:**\n"
    "1.  **DefineAgent**: Creates business glossaries from an *Excel schema file* and a text description.\n"
    "2.  **LineageExtractorAgent**: Analyzes ETL scripts to generate documentation.\n"
    "3.  **DiscoverAgent**: Connects to a *live data source*, discovers its schema, and generates a business glossary. It can connect to three types of sources:\n"
    "    -   **Databricks**: Needs a server hostname, HTTP path, access token\n"
    "    -   **Azure Purview**: Needs a tenant ID, client ID, client secret, and the Purview account name.\n"
    "    -   **Traditional RDBMS** (like PostgreSQL, SQL Server): Needs the database type, host, username, password, and database name.\n\n"
    "**Your Task:**\n"
    "1.  Based on the user's request, determine the primary goal (e.g., 'create from excel', 'analyze script', 'discover from live database').\n"
    "2.  If the goal is to discover from a live database, **you must first identify the specific system** (Databricks, Purview, or a standard database like PostgreSQL).\n"
    "3.  **Crucially, you must ask the user for the specific set of credentials required for that system.** Do NOT ask for a password if they want to connect to Databricks. For example:\n"
    "    -   If they say 'connect to Databricks', you must respond: 'To connect to Databricks, I need the server hostname, the HTTP path from the SQL warehouse, a personal access token, and the name of the catalog you want to analyze.'\n"
    "    -   If they say 'connect to Purview', you must respond: 'To connect to Azure Purview, I need the Tenant ID, Client ID, Client Secret, and the Purview Account Name.'\n"
    "    -   If they say 'connect to our Postgres DB', you must respond: 'To connect to PostgreSQL, I need the host, username, password, and the database name.'\n"
    "4.  Once all required information for a specific task is gathered, route the task to the appropriate agent by responding with *only* the agent's name (e.g., 'DiscoverAgent').\n"
    "5.  If the user is just greeting or their request is vague, ask for clarification on which task they'd like to perform."
    "Do NOT answer any other questions or provide information outside the scope of these agents."
)

# from langchain.chat_models import init_chat_model

master_agent_graph = create_supervisor(
    model=supervisor_llm,
    agents=[define_agent, lineage_extractor_agent, discover_agent],
    prompt=supervisor_system_prompt,
).compile()


# Step 4
# Helper for creating dummy files for testing
def create_dummy_file(
    filename, content="dummy content", directory="supervisor_test_inputs"
):
    os.makedirs(directory, exist_ok=True)
    filepath = os.path.join(directory, filename)
    with open(filepath, "w") as f:
        f.write(content)
    return filepath


if __name__ == "__main__":
    # Clean up for fresh run
    if os.path.exists("supervisor_test_inputs"):
        shutil.rmtree("supervisor_test_inputs")
    if os.path.exists(DEFINE_AI_OUTPUT_DIR):
        shutil.rmtree(DEFINE_AI_OUTPUT_DIR)
    if os.path.exists(LINEAGE_EXTRACTOR_OUTPUT_DIR):
        shutil.rmtree(LINEAGE_EXTRACTOR_OUTPUT_DIR)
    if os.path.exists(DISCOVER_AI_OUTPUT_DIR):
        shutil.rmtree(DISCOVER_AI_OUTPUT_DIR)
    os.makedirs("supervisor_test_inputs", exist_ok=True)
    os.makedirs(DEFINE_AI_OUTPUT_DIR, exist_ok=True)
    os.makedirs(LINEAGE_EXTRACTOR_OUTPUT_DIR, exist_ok=True)
    os.makedirs(DISCOVER_AI_OUTPUT_DIR, exist_ok=True)

    # Conversational Loop
    conversation_history = []

    while True:
        user_input = input(">>> User: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Ending conversation.")
            break

        conversation_history.append(HumanMessage(content=user_input))

        # If the user mentions uploading a file, we simulate it here for the tools to find.
        # In a real UI, the UI would handle the upload and provide the path.
        # For this demo, if user says "using schema.xlsx", we create it.
        # This is a simplification; robust file path handling from text is hard.
        # The supervisor prompt instructs it to ASK for paths if unclear.
        if "schema.xlsx" in user_input.lower():
            create_dummy_file(
                "schema.xlsx",
                "Schema Name,Table Name,Column Name\nSALES,ORDERS,ORDER_ID",
                directory="supervisor_test_inputs",
            )
            print(
                "(Simulated upload of schema.xlsx to supervisor_test_inputs/schema.xlsx)"
            )
        if "etl_script.sql" in user_input.lower():
            create_dummy_file(
                "etl_script.sql",
                "SELECT * FROM users;",
                directory="supervisor_test_inputs",
            )
            print(
                "(Simulated upload of etl_script.sql to supervisor_test_inputs/etl_script.sql)"
            )
        if "glossary.xlsx" in user_input.lower():
            create_dummy_file(
                "glossary.xlsx",
                "Table Name,Column Name,Description\nusers,user_id,User ID",
                directory="supervisor_test_inputs",
            )
            print(
                "(Simulated upload of glossary.xlsx to supervisor_test_inputs/glossary.xlsx)"
            )

        print("\n--- Master Agent Processing ---")
        # The supervisor expects a list of messages
        # For the stream, the input should be a dictionary with a "messages" key
        events = master_agent_graph.stream({"messages": conversation_history})
        ai_response_content = ""  # To store the last user-facing AI message
        final_event_message = None
        print("\n--- Raw Stream Events ---")
        for event_dict in events:
            print(event_dict)

            for agent_name, agent_output in event_dict.items():
                if (
                    isinstance(agent_output, dict)
                    and "messages" in agent_output
                    and agent_output["messages"]
                ):
                    final_event_messages = agent_output["messages"]
        print("--- End Raw Stream Events ---\n")

        if final_event_messages:
            last_msg_in_event = final_event_messages[-1]
            if isinstance(last_msg_in_event, AIMessage):

                if (
                    last_msg_in_event.content not in members
                    and last_msg_in_event.content != "supervisor"
                ):
                    ai_response_content = last_msg_in_event.content

        if ai_response_content:
            print(f"<<< AI: {ai_response_content}")
            conversation_history.append(
                AIMessage(
                    content=ai_response_content,
                    name=last_msg_in_event.name or "supervisor",
                )
            )
        else:
            print(
                "<<< AI: (No explicit user-facing message to display from this turn's events. The supervisor might be routing or an agent is processing.)"
            )
            # If no user-facing message, but supervisor  produced some AIMessage (like a routing instruction),

        print("--- Turn End ---\n")
