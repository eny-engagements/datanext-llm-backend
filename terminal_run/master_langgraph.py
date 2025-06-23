import pandas as pd
import os
import shutil
import asyncio
from langchain_core.tools import tool
from typing import Optional
from etl_documentation import analyse_sp as actual_lineage_extractor_analyse_sp
from etl_documentation import visualise_etl as actual_lineage_extractor_visualise_etl
from etl_documentation import create_etl_documentation_excel as actual_lineage_extractor_create_excel
from etl_documentation import get_glossary as actual_lineage_extractor_get_glossary

from define import read_schema_in_chunks_and_batch as actual_define_read_schema
from define import parallel_description as actual_define_parallel_description
from define import create_business_glossary_excel as actual_define_create_excel

from discover import create_db_engine, get_schemas, get_tables, create_table_overview, create_glossary, generate_csv
import json

DEFINE_AI_OUTPUT_DIR = "C:/Users/TR613EZ/Downloads/define_ai_output/"
LINEAGE_EXTRACTOR_OUTPUT_DIR = "C:/Users/TR613EZ/Downloads/lineage_extractor_output/"
DISCOVER_AI_OUTPUT_DIR = "./output/discovered glossary"
os.makedirs(DEFINE_AI_OUTPUT_DIR, exist_ok=True)
os.makedirs(LINEAGE_EXTRACTOR_OUTPUT_DIR, exist_ok=True)
os.makedirs(DISCOVER_AI_OUTPUT_DIR, exist_ok=True)

@tool
def define_data_model_from_excel(excel_path: str, schema_description: str) -> str:
    """
    Creates a business glossary and data model definitions from an input Excel file
    containing schema information (Schema Name, Table Name, Column Name) and a
    natural language description of the schema.
    The Excel file should have columns like 'Schema Name', 'Table Name', 'Column Name'.
    Returns a summary message with the path to the generated Excel business glossary.
    """
    print(f"--- TOOL: define_data_model_from_excel called with excel_path: {excel_path}, schema_description: '{schema_description}' ---")
    if not os.path.exists(excel_path):
        return f"Error: Input Excel file not found at '{excel_path}'."
    if not schema_description:
        return "Error: Schema description is required."

    try:
        _, schema_info_batches = actual_define_read_schema(excel_path)
        all_glossary_parts = []

        for batch in schema_info_batches:
            glossary_part = actual_define_parallel_description(schema_description, batch, data_dictionary=[])
            all_glossary_parts.append(glossary_part)

        final_glossary_markdown = "  \n".join(all_glossary_parts)

        schema_name_from_path = os.path.basename(excel_path)
        actual_define_create_excel(schema_name_from_path, final_glossary_markdown)

        output_excel_filename = schema_name_from_path.replace(".xlsx", "") + "_with_Business_Glossary.xlsx"
        output_excel_path = os.path.join(DEFINE_AI_OUTPUT_DIR, output_excel_filename)

        return f"Business Glossary generation complete. Markdown is available (if displayed by agent). Excel output saved to: {output_excel_path}"
    except Exception as e:
        return f"Error in define_data_model_from_excel: {e}"

@tool
def extract_etl_lineage_and_documentation(
    script_path: str,
    dialect: str,
    additional_context: Optional[str] = "N/A",
    business_glossary_excel_path: Optional[str] = None
) -> str:
    """
    Analyzes an ETL script (SQL or text file) to generate documentation, a data lineage diagram,
    and an Excel report.
    Required inputs are the path to the script file and the SQL dialect (e.g., 'T-SQL', 'Spark SQL').
    Optional inputs include additional natural language context and the path to a business glossary Excel file.
    Returns a summary message with paths to the generated lineage diagram and Excel documentation.
    """
    print(f"--- TOOL: extract_etl_lineage_and_documentation called with script_path: {script_path}, dialect: {dialect} ---")
    if not os.path.exists(script_path):
        return f"Error: Input script file not found at '{script_path}'."
    if not dialect:
        return "Error: SQL dialect is required (e.g., 'T-SQL', 'Spark SQL')."

    full_context = additional_context
    if business_glossary_excel_path:
        if not os.path.exists(business_glossary_excel_path):
            print(f"Warning: Glossary file '{business_glossary_excel_path}' not found, proceeding without it.")
        else:
            try:
                with open(script_path, 'r', encoding='utf-8') as f:
                    script_content = f.read()
                glossary_context_str = actual_lineage_extractor_get_glossary(script_content, business_glossary_excel_path)
                full_context += f"\n\nBusiness Glossary (from {os.path.basename(business_glossary_excel_path)}):\n{glossary_context_str}"
            except Exception as e:
                print(f"Warning: Could not process glossary file '{business_glossary_excel_path}': {e}. Proceeding without it.")

    try:
        report_name = os.path.basename(script_path)
        with open(script_path, 'r', encoding='utf-8') as f:
            script_content_for_analysis = f.read()

        df_analysis, df_filled_analysis = actual_lineage_extractor_analyse_sp(
            script_content_for_analysis,
            full_context,
            dialect
        )
        diagram_path = actual_lineage_extractor_visualise_etl(script_content_for_analysis, report_name, df_analysis)
        actual_lineage_extractor_create_excel(df_filled_analysis, report_name)
        output_excel_filename = report_name.replace(".txt", " ETL Documentation.xlsx").replace(".sql", " ETL Documentation.xlsx")
        output_excel_path = os.path.join(LINEAGE_EXTRACTOR_OUTPUT_DIR, output_excel_filename)

        # The markdown table (df_analysis.to_markdown()) could be returned if the agent is prompted to display it.
        return (f"ETL Lineage and Documentation generation complete. "
                f"Lineage Diagram saved to: {diagram_path}. "
                f"Excel documentation saved to: {output_excel_path}.")
    except Exception as e:
        return f"Error in extract_etl_lineage_and_documentation: {e}"
    
@tool
def discover_database_and_generate_glossary(db_type: str, host: Optional[str] = None, username: Optional[str] = None, password: Optional[str] = None, database: Optional[str] = None, port: Optional[str] = None, service: Optional[str] = None, authentication: Optional[str] = None) -> str:
    """
    Connects to a live database (PostgreSQL, MySQL, Microsoft SQL Server, OracleDB, SQLite),
    discovers all its non-system schemas and tables, and generates a comprehensive business glossary.
    Requires connection details appropriate for the specified db_type.
    Returns a summary message with the path to the generated CSV glossary.
    """
    print(f"--- TOOL: discover_database_and_generate_glossary called for db_type: {db_type} ---")
    config = {k: v for k, v in locals().items() if v is not None}
    config['rdbms'] = config.pop('db_type')

    engine, error = create_db_engine(config)
    if error:
        return f"Connection Failed: {error}"

    try:
        all_schemas, error = get_schemas(engine)
        if error: return f"Schema Discovery Failed: {error}"
        print(f"Discovered schemas: {all_schemas}")

        # In a real tool, we might let the user select schemas/tables.
        # Here, we simplify by processing all discovered non-system schemas.
        # We need to construct the input for `create_table_overview`
        selected_schemas_per_connection = {"live_connection": all_schemas}
        selected_tables_per_schema = {}
        for schema in all_schemas:
            tables, tbl_error = get_tables(engine, schema)
            if tbl_error: continue
            selected_tables_per_schema[schema] = tables

        overview = create_table_overview(
            {"live_connection": {"engine": engine}},
            selected_schemas_per_connection,
            selected_tables_per_schema
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

# Step 2
from langchain_openai import AzureChatOpenAI
from langgraph.prebuilt import create_react_agent


agent_llm = AzureChatOpenAI(
    azure_deployment="gpt-4o",
    openai_api_version="2024-02-15-preview",
    api_key="901cce4a4b4045b39727bf1ce0e36219",
    azure_endpoint="https://aifordata-azureopenai.openai.azure.com/",
    temperature=0,
    max_tokens=4096
    )

define_agent = create_react_agent(
    model=agent_llm, 
    tools=[define_data_model_from_excel],
    prompt=(
        "You are the DefineAgent, a specialist in creating business glossaries and "
        "data model definitions from Excel schema files and descriptions provided by the user. "
        "If the user provides an Excel file path and a schema description, use the "
        "'define_data_model_from_excel' tool. Ensure you have both pieces of information. "
        "If an Excel file path or schema description is missing, ask the user to provide it clearly."
    ),
    name="DefineAgent"
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
    name="LineageExtractorAgent"
)

discover_agent = create_react_agent(
    model=agent_llm,
    tools=[discover_database_and_generate_glossary],
    prompt=(
        "You are the DiscoverAgent, a specialist in connecting to live databases, discovering their "
        "metadata, and generating a business glossary. To use the 'discover_database_and_generate_glossary' "
        "tool, you need the database connection details.\n\n"
        "**Your primary task is to look at the user's messages in the conversation history to find these details.**\n"
        "Extract the following arguments for the tool from the conversation:\n"
        "- db_type (e.g., PostgreSQL, MySQL, etc.)\n"
        "- host\n"
        "- username\n"
        "- password\n"
        "- database (the database name)\n\n"
        "Once you have gathered all necessary arguments from the conversation history, call the "
        "'discover_database_and_generate_glossary' tool with all the extracted arguments."
    ),
    name="DiscoverAgent"
)

from langgraph_supervisor import create_supervisor
from langchain_core.messages import HumanMessage, AIMessage

supervisor_llm = AzureChatOpenAI(model="gpt-4o", temperature=0,    
                                 openai_api_version="2024-02-15-preview",
    api_key="901cce4a4b4045b39727bf1ce0e36219",
     azure_endpoint="https://aifordata-azureopenai.openai.azure.com/") 

members = ["DefineAgent", "LineageExtractorAgent", "DiscoverAgent"]

#Write a function to process yaml

# Create the supervisor
# supervisor_system_prompt = (
#     "You are the Master AI Agent. Your role is to manage a conversation between a user "
#     "and a team of specialized AI agents: 'DefineAgent' and 'LineageExtractorAgent'.\n"
#     "1. DefineAgent: Creates business glossaries and data models from an Excel schema file and a schema description.\n"
#     "   - Needs: Excel file path, schema description (text).\n"
#     "2. LineageExtractorAgent: Analyzes ETL scripts to generate documentation and lineage diagrams.\n"
#     "   - Needs: Script file path, SQL dialect (e.g., 'T-SQL', 'Spark SQL').\n"
#     "   - Optional: Business glossary Excel file path, additional text context.\n\n"
#     "Based on the user's request and the conversation history:\n"
#     "- First, determine which agent is best suited to handle the request.\n"
#     "- Before routing to an agent, check if all *required* inputs for that agent are clearly available from the user's messages or previous context. "
#     "  You can infer file paths if the user mentions them like 'use script.sql' or 'the excel is schema_data.xlsx'.\n"
#     "- If required inputs (like file paths, schema descriptions, or SQL dialect) are missing or unclear, "
#     "  you MUST ask the user for the specific missing information. For example: 'To help you with that, I need the path to your Excel schema file and a brief description of the schema.' or 'Please provide the ETL script file and specify its SQL dialect (e.g., T-SQL or Spark SQL).'\n"
#     "- Once all required information for an agent is gathered, route the task to that agent by responding with the agent's name (e.g., 'DefineAgent' or 'LineageExtractorAgent').\n"
#     "- If the user's request is too vague or doesn't seem to fit either agent, ask for clarification about what they want to achieve.\n"
#     "- If an agent completes its task or returns an error, present that information to the user and await their next instruction.\n"
#     "Do not call the tools yourself; delegate to the agents."
#     ""
# )
# supervisor_system_prompt = (
#     "You are the Master AI Agent. Your role is to manage a conversation between a user "
#     "and a team of specialized AI agents: 'DefineAgent', 'LineageExtractorAgent', and 'DiscoverAgent'.\n"
#     "1. DefineAgent: Creates business glossaries from an *Excel schema file* and a schema description.\n"
#     "   - Needs: Excel file path, schema description (text).\n"
#     "2. LineageExtractorAgent: Analyzes ETL scripts to generate documentation and lineage diagrams.\n"
#     "   - Needs: Script file path, SQL dialect (e.g., 'T-SQL', 'Spark SQL').\n"
#     "3. DiscoverAgent: Connects to a *live database*, discovers its schema, and generates a business glossary.\n"
#     "   - Needs: Database connection details (type, host, credentials, database name, etc.).\n\n"
#     "Based on the user's request, first determine which agent is best suited. Distinguish carefully:\n"
#     "- If the user mentions an *Excel file* for creating a glossary, use 'DefineAgent'.\n"
#     "- If the user mentions connecting to a *live database* to generate a glossary, use 'DiscoverAgent'.\n"
#     "- If the user mentions an *ETL script* or *stored procedure*, use 'LineageExtractorAgent'.\n\n"
#     "Before routing to an agent, you MUST ask the user for any specific missing *required* information. For example:\n"
#     "- For DiscoverAgent: 'To connect to your database, I need the type (e.g., PostgreSQL), host, username, password, and database name.'\n"
#     "- For LineageExtractorAgent: 'Please provide the ETL script file and specify its SQL dialect. Optionally use the business glossary generated by DefineAgent.'\n"
#     "- For DefineAgent: 'Please provide the Excel schema file and a brief description.'\n\n"
#     "Once all required information is gathered, route the task to the appropriate agent by responding with only the agent's name. "
#     "If the user is just greeting or their request is vague, ask for clarification on which task they'd like to perform."
#     "DONOT answer any other questions like what is the weather, etc or provide information outside the scope of these agents. "
#     "If the user asks for something unrelated, politely redirect them to the task at hand.\n"
# )
supervisor_system_prompt = (
    "You are the Master AI Agent, a friendly and professional conversational assistant. Your role is to manage a conversation between a user "
    "and a team of specialized AI agents: 'DefineAgent', 'LineageExtractorAgent', and 'DiscoverAgent'. Your primary goal is to gather all necessary information from the user step-by-step before routing to an agent to execute a tool.\n\n"
    
    "## High-Level Task Routing ##\n"
    "First, understand the user's main goal:\n"
    "- If they want to **create a business glossary** or **define a data model**, you must follow the 'DefineAgent Conversational Flow'.\n"
    "- If they want to **analyze an ETL script** or **see data lineage**, you must ask for the script path and dialect, then route to `LineageExtractorAgent`.\n"
    "- If they want to **connect to a live database**, you must ask for connection details, then route to `DiscoverAgent`.\n"
    "- If the request is a simple greeting or vague, ask for clarification on which task they want to perform.\n\n"
    
    "--- START: DefineAgent Conversational Flow Rulebook ---\n"
    "If the user wants to create a business glossary, you MUST follow these exact steps:\n\n"
    "**Phase 1: Initial Context Gathering**\n"
    "1. Your first question MUST be to ask the user for their business sector or industry (e.g., Finance, Healthcare, Retail).\n"
    "2. After they provide the industry, your next question MUST be to ask for their data source. Give them three options: '1. Connect to a live database', '2. Upload an Excel file', or '3. Use metadata from a previous Discover.ai run'.\n\n"
    
    "**Phase 2: Data Source Workflow**\n"
    "**--> If user chooses 'Connect to a live database':**\n"
    "   - Respond by saying you will hand them over to the DiscoverAgent. Then, route to `DiscoverAgent`.\n\n"

    "**--> If user chooses 'Upload an Excel file':**\n"
    "   - **Step Excel-1:** Ask the user to provide the path to their Excel file.\n"
    "   - **Step Excel-2:** After they provide the path, ask for a natural language schema description.\n"
    "   - **Step Excel-3:** Once you have both the file path and the description, you can route to the `DefineAgent`. Before routing, confirm with the user, like: 'Great, I have the file path and description. I will now ask the DefineAgent to process it.'\n\n"

    "**Phase 3: After Tool Execution**\n"
    "When an agent tool call is finished and you receive the result, present the summary to the user. Then, you MUST ask what they want to do next. For a generated glossary, offer options like:\n"
    "'Would you like to: 1. View the full glossary here? 2. Download the glossary as a file? 3. Save the glossary for future reference?' (Note: Treat these as conversational follow-ups for now).\n\n"
    
    "**Error Handling:**\n"
    "If a tool call fails, present the error clearly to the user and suggest a possible solution, like checking the file path or format.\n"
    "--- END: DefineAgent Conversational Flow Rulebook ---\n\n"


    "--- START: LineageExtractorAgent Conversational Flow Rulebook ---\n"
    "If the user wants to analyze an ETL script, you MUST follow these exact steps:\n\n"
    
    "**Phase 1: Initial Context Gathering**\n"
    "1. Just like with DefineAgent, your first question MUST be to ask the user for their business sector or industry. This helps contextualize the analysis.\n"
    "2. After they provide the industry, your next question MUST be to ask for the required inputs. Clearly state what you need. For example: 'Awesome! To get started, please provide the stored procedure file and select the SQL dialect (e.g., T-SQL, Spark SQL). You can also provide additional context or a business glossary file.'\n\n"
    
    "**Phase 2: Input Handling**\n"
    "- **Required Inputs:** Do not proceed until you have the **script file path** and the **SQL dialect**.\n"
    "- **Optional Glossary:** If the user wants to use a glossary:\n"
    "  - If a glossary was generated earlier in THIS conversation, remember its file path. Proactively ask: 'Would you like to use the business glossary we just created for [previous task]?'\n"
    "  - If they provide a new glossary file path, accept it.\n"
    "  - If they don't provide one and one isn't in the history, that's okay, it's optional.\n"
    "- Once you have the required inputs AND any optional glossary path, confirm ALL details with the user before proceeding: 'I have all the information needed: Script: [path], Dialect: [dialect], Glossary: [path or None]. Proceeding with analysis...' and then route the task to the `LineageExtractorAgent`.\n\n"
    
    "**Phase 3: Presenting Results**\n"
    "1. After the `LineageExtractorAgent` tool finishes successfully, announce it: 'Lineage analysis is done!'\n"
    "2. You MUST then ask the user what they want to do next by offering these options: 'Would you like to: 1. View the full report here? 2. Download the Excel file? 3. Save the report in your workspace for later? 4. All?' (Note: Treat 'View' and 'Save' as conversational follow-ups for now).\n\n"
    
    "**Error & Edge Case Handling:**\n"
    "- **Invalid Script File:** If the agent's tool returns an error about the file, inform the user: 'The file seems invalid or unsupported. Please upload a valid stored procedure file in .txt or .sql format.'\n"
    "- **Invalid Glossary File:** If the tool reports an error with the glossary, say: 'The glossary file appears incorrectly formatted. Please ensure it has at least 'Schema Name', 'Table Name', and 'Column Name' columns.'\n"
    "- **User has no files:** If the user says they don't have a script, respond gracefully: 'No worries! Whenever you have a script ready for analysis, I'll be here to help.'\n"
    "--- END: LineageExtractorAgent Conversational Flow Rulebook ---\n"


    "--- START: DiscoverAgent Conversational Flow Rulebook ---\n"
    "If the user wants to extract metadata using the DiscoverAgent, you MUST follow these exact steps:\n\n"
    
    "**Phase 1: Initial Context Gathering**\n"
    "1. Your first question MUST be to ask the user for their business sector or industry (e.g., Finance, Healthcare).\n"
    "2. After they provide the industry, your next question MUST be to ask for their data source. Say: 'Great! To get started, please provide your data source. You can either: 1. Connect your database, or 2. Upload an Excel file containing your metadata. Which option would you prefer?'\n\n"
    
    "**Phase 2: Data Source Specific Workflow**\n"
    "**--> If user chooses 'Connect your database':**\n"
    "   - Ask for all necessary connection details (Database Type, Host, Username, Password, Database Name).\n"
    "   - Once you have all details, confirm and route to the `DiscoverAgent` for processing.\n\n"

    "**--> If user chooses 'Upload an Excel file':**\n"
    "   - **Step Excel-1:** Ask the user to provide the path to their Excel file.\n"
    "   - **Step Excel-2:** After they provide the path, acknowledge it and then route the task to the `DefineAgent`. **Important:** The `DiscoverAgent` tool is for live databases only. The task of processing a metadata Excel file belongs to the `DefineAgent`. You should say: 'Thanks for providing the Excel file. I will now route you to the `DefineAgent`, which specializes in handling Excel-based schemas, to create your business glossary.' Then, route to `DefineAgent`.\n\n"
    
    "**Phase 3: Presenting Results (for DB connection)**\n"
    "1. After the `DiscoverAgent` tool finishes successfully, announce it: 'Metadata extraction from your database completed successfully!'\n"
    "2. Ask the user what they want to do next with the generated metadata: 'Would you like to: 1. View the full metadata here? 2. Download the metadata as a CSV file? ...'\n\n"
    ## For discover, download option should not regenrate the file
    ## For saving, allow the user to rename the file
    "**Error & Edge Case Handling:**\n"
    "- **DB Connection Fails:** If the `DiscoverAgent` tool returns an error, inform the user and suggest they check credentials or try the Excel upload method as an alternative.\n"
    "- **User has no data source:** If the user says they don't have a database or an Excel file, respond gracefully: 'No worries! To extract metadata, I need a data source. Would you like me to provide a sample Excel template you can fill out and upload later?'\n"
    "--- END: DiscoverAgent Conversational Flow Rulebook ---\n"
    "**Final Instruction:** Your job is to have the conversation. Once all information is collected for a specific tool, and only then, route to the correct agent ('DefineAgent', 'LineageExtractorAgent', 'DiscoverAgent') so they can execute their tool."
)


# from langchain.chat_models import init_chat_model

master_agent_graph = create_supervisor(
    model=supervisor_llm, 
    agents=[define_agent, lineage_extractor_agent, discover_agent],
    prompt=supervisor_system_prompt,
).compile()

#Step 4
# Helper for creating dummy files for testing
def create_dummy_file(filename, content="dummy content", directory="supervisor_test_inputs"):
    os.makedirs(directory, exist_ok=True)
    filepath = os.path.join(directory, filename)
    with open(filepath, "w") as f:
        f.write(content)
    return filepath

# Clean up for fresh run 
if os.path.exists("supervisor_test_inputs"): shutil.rmtree("supervisor_test_inputs")
if os.path.exists(DEFINE_AI_OUTPUT_DIR): shutil.rmtree(DEFINE_AI_OUTPUT_DIR)
if os.path.exists(LINEAGE_EXTRACTOR_OUTPUT_DIR): shutil.rmtree(LINEAGE_EXTRACTOR_OUTPUT_DIR)
if os.path.exists(DISCOVER_AI_OUTPUT_DIR): shutil.rmtree(DISCOVER_AI_OUTPUT_DIR)
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
        create_dummy_file("schema.xlsx", "Schema Name,Table Name,Column Name\nSALES,ORDERS,ORDER_ID", directory="supervisor_test_inputs")
        print("(Simulated upload of schema.xlsx to supervisor_test_inputs/schema.xlsx)")
    if "etl_script.sql" in user_input.lower():
        create_dummy_file("etl_script.sql", "SELECT * FROM users;", directory="supervisor_test_inputs")
        print("(Simulated upload of etl_script.sql to supervisor_test_inputs/etl_script.sql)")
    if "glossary.xlsx" in user_input.lower():
         create_dummy_file("glossary.xlsx", "Table Name,Column Name,Description\nusers,user_id,User ID", directory="supervisor_test_inputs")
         print("(Simulated upload of glossary.xlsx to supervisor_test_inputs/glossary.xlsx)")


    print("\n--- Master Agent Processing ---")
    # The supervisor expects a list of messages
    # For the stream, the input should be a dictionary with a "messages" key
    events = master_agent_graph.stream({"messages": conversation_history})
    ai_response_content = "" # To store the last user-facing AI message
    final_event_message = None
    print("\n--- Raw Stream Events ---")
    for event_dict in events:
        print(event_dict) 

  
        for agent_name, agent_output in event_dict.items(): 
            if isinstance(agent_output, dict) and "messages" in agent_output and agent_output["messages"]:
                final_event_messages = agent_output["messages"] 
    print("--- End Raw Stream Events ---\n")

    if final_event_messages: 
        last_msg_in_event = final_event_messages[-1]
        if isinstance(last_msg_in_event, AIMessage):
            
            if last_msg_in_event.content not in members and last_msg_in_event.content != "supervisor":
                ai_response_content = last_msg_in_event.content

    if ai_response_content:
        print(f"<<< AI: {ai_response_content}")
        conversation_history.append(AIMessage(content=ai_response_content, name=last_msg_in_event.name or "supervisor"))
    else:
        print("<<< AI: (No explicit user-facing message to display from this turn's events. The supervisor might be routing or an agent is processing.)")
        #If no user-facing message, but supervisor  produced some AIMessage (like a routing instruction),
        

    print("--- Turn End ---\n")