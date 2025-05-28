import streamlit as st
 
from sqlalchemy import create_engine, inspect, select, text, Table, MetaData
from sqlalchemy.schema import CreateTable
from sqlalchemy_utils import database_exists
from sqlalchemy.engine.url import URL
from sqlalchemy.exc import SQLAlchemyError
from urllib.parse import quote_plus
 
import os
from typing import Any, Set, Dict
import json
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
 
from ydata_profiling import ProfileReport
 
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain_openai import AzureChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
 
 
INITIAL_PROFILE_JSON_PATH = './output/data quality/data_profile_report.json'
PROFILE_JSON_PATH = './output/data quality/cleaned_profile_report.json'
EMBEDDINGS_FILE_PATH = "./input/data_quality.pkl"
 
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
 
embeddings = HuggingFaceEmbeddings(model_name="./all-mpnet-base-v2/", model_kwargs={'trust_remote_code': True})
 
if os.path.exists(EMBEDDINGS_FILE_PATH):
    with open(EMBEDDINGS_FILE_PATH, 'rb') as f:
        vector_store = pickle.load(f)
 
 
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
 
def get_table_names(_engine):
    """Gets a list of table names in the format schema.table from the database."""
    if _engine is None:
        return []
    try:
        inspector = inspect(_engine)
        schemas = inspector.get_schema_names()
        tables = []
        for schema in schemas:
            schema_tables = inspector.get_table_names(schema=schema)
            tables.extend([f"{schema}.{table}" for table in schema_tables if "validated_" not in table and "quarantined_" not in table])
        return tables
    except SQLAlchemyError as e:
        st.error(f"Error fetching table names: {e}")
        return []
    except Exception as e:
        st.error(f"An unexpected error occurred fetching tables: {e}")
        return []
 
def fetch_full_table(engine, table_name):
    """Fetches the ENTIRE content of a specified table."""
    query = ""
    if not table_name or engine is None:
        return pd.DataFrame()
    # st.write(f"Attempting to fetch full table: `{table_name}`...") # User feedback
    try:
        # Query without LIMIT to get all rows
        # query = text(f'SELECT * FROM `{table_name}`') # Use backticks for safety
        # with engine.connect() as connection:
        #     df = pd.read_sql_query(query, connection)
        # metadata = MetaData()
        # table_reflection = Table(table_name, metadata, autoload_with=engine)
        # stmt = select([table_reflection])
        # return pd.read_sql(stmt, engine)
        # st.write(f"Successfully fetched `{table_name}`.") # More feedback
        # return df
        if engine.dialect == "mssql":
            query = f"SELECT TOP 5 * FROM dbo.{table_name}"
        
        df = pd.read_sql(query, engine)
        return df
    # except SQLAlchemyError as e:
    #     st.error(f"Error fetching full data for table '{table_name}': {e}")
    #     return pd.DataFrame() # Return empty DataFrame on error
    except Exception as e:
        st.error(f"An unexpected error occurred fetching full data for '{table_name}': {e}")
        return pd.DataFrame()
    
def fetch_table_data(engine, table_name, n=5):
    try:
        if engine.name == 'mssql':
            query = f"SELECT TOP {n} * FROM {table_name}"
        elif engine.name == 'oracle':
            query = f"SELECT * FROM {table_name} FETCH FIRST {n} ROWS ONLY;"
        elif engine.name == 'sqlite':
            query = f"SELECT * FROM {table_name} LIMIT {n}"
        else:
            query = f"SELECT * FROM {table_name} LIMIT {n}"  
        df = pd.read_sql(query, engine)
        return df
    except Exception as e:
        st.error(f"An unexpected error occurred fetching full data for '{table_name}': {e}")
        return pd.DataFrame()
    
def get_business_glossary(file_path):
    dd_df = pd.read_excel(file_path, engine='openpyxl')
    dd_df['Schema Table'] = dd_df['Schema'] + '.' + dd_df['Table']
    
    # Create the combined dictionary
    glossary = dd_df.groupby('Schema Table').apply(
        lambda group: {
            'description': group['Table Description'].iloc[0],
            'column_descriptions': '\n'.join(f"{row['Column']} - {row['Definition']}" for _, row in group.iterrows())
        }
    ).to_dict()
    
    return glossary
 
def clean_profile_report(profile_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Removes specified keys from a ydata-profiling report dictionary (in-place)
    to reduce size, primarily for LLM context windows.
 
    Args:
        profile_data: The loaded ydata-profiling report as a Python dictionary.
 
    Returns:
        The modified profile_data dictionary with specified keys removed.
        Note: The modification happens in-place, but the dictionary is also returned.
    """
 
    # Keys identified as often large or less critical for DQ rule generation context
    top_level_keys_to_remove: Set[str] = {
        'scatter', 'package', 'sample', 'analysis', 'time_index_analysis',
    }
 
    common_variable_keys_to_remove: Set[str] = {
        'block_alias_values', 'block_alias_counts', 'n_block_alias',
        'block_alias_char_counts', 'histogram_length', 'length_histogram', 'histogram',
        'memory_size', 'n_characters', 'n_characters_distinct', 'character_counts',
        'category_alias_values', 'script_counts', 'n_scripts',
        'script_char_counts', 'category_alias_counts', 'n_category',
        'category_alias_char_counts', 'word_counts', 'value_counts_index_sorted',
    }
 
    conditional_variable_key_to_remove: str = 'value_counts_without_nan'
    conditional_types: Set[str] = {'Numeric', 'DateTime', 'Text', 'URL', 'Path', 'Unsupported'}
 
    # 2. Remove top-level keys
    # print("Removing top-level keys...")
    keys_removed_top = []
    for key in top_level_keys_to_remove:
        if profile_data.pop(key, None) is not None: # Use pop for safety
            keys_removed_top.append(key)
    # if keys_removed_top:
    #     print(f"   - Removed top-level keys: {', '.join(keys_removed_top)}")
    # else:
    #     print("   - No specified top-level keys found to remove.")
 
 
    # 3. Process variables section
    if 'variables' in profile_data and isinstance(profile_data['variables'], dict):
        # print("Processing variables...")
        variables = profile_data['variables']
        for var_name, var_details in variables.items():
            if isinstance(var_details, dict): # Ensure it's a dictionary
                # print(f"   Processing variable: {var_name}") # Can be verbose, optional
 
                # 3a. Remove common keys from this variable
                keys_removed_common = []
                for key in common_variable_keys_to_remove:
                    if var_details.pop(key, None) is not None: # Use pop for safety
                        keys_removed_common.append(key)
                # if keys_removed_common:
                #     print(f"     - Removed common keys: {', '.join(keys_removed_common)}") # Verbose
 
                # 3b. Conditionally remove specific key based on type
                var_type = var_details.get('type') # Use .get() for safety
                if var_type in conditional_types:
                    if var_details.pop(conditional_variable_key_to_remove, None) is not None: # Use pop
                        # print(f"     - Removed conditional key '{conditional_variable_key_to_remove}' (type: {var_type})") # Verbose
                        pass # Silenced verbose output
 
            # else:
            #    print(f"   Skipping variable '{var_name}': Value is not a dictionary.")
    # else:
    #     print("No 'variables' section found or it's not a dictionary.")
 
    # print("Cleaning complete.")
    return profile_data
 
def generate_profile(df):
    profile = ProfileReport(
        df,
        title="Data Profiling Report for My Dataset",
        explorative=True,
    )
 
    report_json = profile.to_json()
 
    try:
        with open(INITIAL_PROFILE_JSON_PATH, 'w') as f:
            f.write(report_json)
    except Exception as e:
        print(f"Error saving JSON to file: {e}")
 
    try:
        with open(INITIAL_PROFILE_JSON_PATH, "r") as f:
            profile_data = json.load(f)
 
        profile_data = clean_profile_report(profile_data)

        with open(PROFILE_JSON_PATH, "w") as f:
            json.dump(profile_data, f)
        # return profile_data
    except Exception as e:
        st.error(e)
 
def display_value(value):
    """Formats values nicely for display."""
    if isinstance(value, float):
        if 0 < abs(value) < 1:
            return f"{value:.2%}"
        else:
            return f"{value:,.2f}"
    if isinstance(value, (int, str)):
        return value
    return str(value)

def export_html(df):
    profile = ProfileReport(
        df,
        title="Data Profiling Report for My Dataset",
        explorative=True,
    )
    profile_html = profile.to_html()


 
def create_rules(profile, context="N/A"):
    retrieved_rules = []
    profile = str(profile)

    generate_vector_query_template = PromptTemplate.from_template(
    """
    You are an expert Data Quality Analyst AI Agent. Your goal is to help users find relevant Data Quality (DQ) rules for their
    specific dataset.
 
    You have access to the following information about the user's dataset:
    1.  **Data Demographics Summary:** Basic statistics, data types, missing values, and value examples.
    2.  **Data Profile Context:** General information derived from a detailed profiling report.
 
    **Context about the DQ Rules Library:**
    The user has a comprehensive vector store containing approximately 8,000 diverse Data Quality rules. This library can be
    searched using natural language queries to find rules relevant to specific data characteristics or potential issues.
 
    **Your Task:**
    Based *only* on the provided Data Profile and Additional Context below, generate a list of 5-10 specific, natural language
    **search queries**. These queries should be designed to effectively search the DQ rules vector store and retrieve rules that
    address the potential needs, sector (insurance, banking, manufacturing, etc) or characteristics highlighted in the dataset's demographics and profile. The search queries should
    not contain codified column names within it. The additional context might contain a data dictionary in case of codified column
    names. Refer to it in generating the queries. Please note that the DQ rules vector store primarily consist of DQ rules relevant
    to the insurance sector but many of the rules provide checks that are consistent accross all sectors.
 
    Focus on generating queries related to:
    * Data types observed (e.g., 'date validation rules', 'integer range checks')
    * Missing value patterns ('handling missing patient IDs', 'rules for optional fields')
    * Value distributions or specific values noted ('rules for 'SEX' column values', 'checking age distribution outliers')
    * Potential inconsistencies suggested by the data ('query for caste name consistency', 'finding rules for category code mapping')
    * Specific column names mentioned in the demographics.
 
    Data Profile:
    {profile}

    Additional Context for the Data:
    {context}
 
    Output Format:
    Provide ONLY a comma separated list of strings, where each string is a natural language search query. Example:
    'query for validating discharge dates against surgery dates', 'rules for ensuring validity of age data', 'checking consistency
    between category code and category name'
    """
    )

    # rule_template = PromptTemplate.from_template(
    # """ 
    # PERSONA:

    # You are an exceptionally skilled Senior Data Quality (DQ) Analyst AI, 
    # expert in interpreting data profiles (ydata-profiling), understanding DQ dimensions (covering dimensions like Uniqueness, Validity,
    # Accuracy, Consistency, Timeliness, Completeness, etc.), 
    # and translating statistical findings into precise, actionable, 
    # row-level validation rules suitable for automated checks in SQL or Python/pandas. 
    # You prioritize clarity, specificity, and evidence-based reasoning.  
    # You are also proficient in SQL and Python/pandas, enabling you to implement these rules effectively.

    # CONTEXT:

    # We're automating DQ rule suggestions. You are given a summary of a dataset's profile (DATASET_PROFILE_CONTEXT) and a list of general DQ rules (RETRIEVED_DQ_RULES) retrieved from our enterprise library via vector search. These retrieved rules are examples only, illustrating the types of validation logic (range checks, format validation, set membership, null checks, uniqueness, consistency) and the expected structure/phrasing for rules within our framework. They DO NOT contain logic specific to the current dataset.

    # OBJECTIVE:

    # Your primary goal is to generate a set of Data Quality rules defining specific, testable pass/fail conditions for individual rows of the dataset described in DATASET_PROFILE_CONTEXT. These rules must be directly derived from the profile's statistics, observed values, and alerts. They should be immediately usable for implementing automated data validation scripts (e.g., in SQL WHERE clauses or pandas boolean masks), moving significantly beyond just summarizing profile alerts. Use the RETRIEVED_DQ_RULES solely for understanding how to structure validation logic and phrase the final rules clearly.

    # INPUTS:

    # DATASET_PROFILE_CONTEXT: Structured summary from ydata-profiling, including:

    # Dataset Overview (rows, columns, missing %, duplicates %)
    # Variable Details (Column Name, Detected Type, Stats [min, max, mean, unique count], Missing %, Zeros %, Negatives %, Alerts [High Cardinality, Skewness, Uniqueness issues, etc.])
    # Correlations/Missing Patterns (if provided).
    # RETRIEVED_DQ_RULES: List of general natural language DQ rules from library (e.g., "Effective date must be before expiration date", "Postal code must match 5-digit format", "Status must be in approved list [...]", "Claim amount must be > 0"). These are for structural/type inspiration ONLY.

    # INSTRUCTIONS (Chain of Thought & Rule Generation):

    # Follow these steps meticulously:

    # Understand Goal: Generate specific, row-level validation rules (pass/fail conditions) based on the profile, inspired structurally by retrieved rules.
    # Analyze DATASET_PROFILE_CONTEXT: Systematically review overview, variables, and alerts. For each column, identify potential needs for validation based on its type, stats, observed values (min/max/unique counts), and alerts. Focus on translating observations into potential checks.
    # Use RETRIEVED_DQ_RULES for Structural Insight: Review retrieved rules to understand how checks for ranges, formats, sets, nullity, uniqueness, or consistency are typically expressed. Apply these patterns to the specific findings from the profile. DO NOT: Copy rules, slightly modify rules, assume business logic from retrieved rules, or use them if they don't match a type of validation suggested by the profile.
    # Reasoning (Chain of Thought - Explicitly Show This): For each validation need identified in Step 2:
    # Observation: State the finding from DATASET_PROFILE_CONTEXT (e.g., "Column 'Age' (Numeric): min=0, max=115, 0.5% missing.").
    # Potential Validation Need & Type: Interpret the need for a specific check (e.g., "Need to ensure Age is not missing and stays within the observed, reasonable range. Requires a Non-Null Check and a Range Check.").
    # Relevant DQ Dimension(s): (e.g., Completeness, Validity).
    # Inspiration (Structure/Type): (e.g., "Retrieved rules show examples of range checks (>0, BETWEEN X and Y) and null checks, confirming these validation patterns.").
    # Rule Formulation (Define Testable Condition): Draft a specific DQ rule defining a clear pass/fail condition for a row. Use imperative language. Infer reasonable constraints from profile data where appropriate, clearly stating the basis for inference.
    # For Numeric: Propose MUST BE >= {{min}} or MUST BE BETWEEN {{min}} AND {{max}} (using observed min/max if they seem valid, otherwise propose standard bounds like >= 0). Note inference.
    # For Categorical: If few unique values observed (e.g., 'A', 'B', 'C'), propose MUST BE ONE OF ('A', 'B', 'C'). Note inference based on observed values.
    # For Text/IDs: If format is suggested (e.g., always 5 digits, looks like email), propose MUST MATCH REGEX '...'. If uniqueness alert exists, propose MUST BE UNIQUE.
    # For Dates: Propose MUST BE a valid date [in YYYY-MM-DD format] or MUST NOT BE NULL or MUST BE ON OR BEFORE {{current_date}} (if future dates observed).
    # For Missing % Alert: Propose MUST NOT BE NULL or MUST NOT BE empty.
    # Justification: Briefly explain why the rule/condition is needed based on the profile observation/inference (e.g., "Ensures age validity based on observed range", "Restricts status to observed values, requires business confirmation", "Addresses non-unique ID alert").
    # Synthesize & Refine Rules:
    # Consolidate rules. Ensure they define testable, row-level conditions.
    # Prioritize rules that directly translate profile stats/alerts into checks (Nullity, Range, Format, Set Membership, Uniqueness, Consistency) over purely descriptive statistical observations.
    # Ensure clear, imperative phrasing (e.g., "MUST BE", "MUST NOT BE NULL", "MUST MATCH REGEX", "MUST BE ONE OF").
    # Tag each rule with its primary DQ Dimension.
    # OUTPUT FORMAT:

    # Present your response in two distinct sections:

    # Section 1: Chain of Thought Reasoning
    # Provide the detailed step-by-step reasoning process (Instruction 4) for each generated rule, clearly showing the link from profile observation to the specific validation condition formulation.

    # Section 2: Generated Data Quality Rules (Row-Level Validation Conditions)

    # Provide a clean, numbered list of the final generated DQ rules.
    # Each rule must clearly state a testable condition using imperative language.
    # Prefix each rule with its primary DQ Dimension tag (e.g., [Validity]).
    # Example Rule Formulations:
    # [Completeness] Column 'Order_ID' MUST NOT BE NULL.
    # [Validity] Column 'Order_Amount' MUST BE GREATER THAN OR EQUAL TO 0.
    # [Validity] Column 'Product_Code' MUST MATCH REGEX '^[A-Z]{{3}}-[0-9]{{5}}$'. (Based on inferred pattern)
    # [Validity] Column 'Ship_Status' MUST BE ONE OF ('Shipped', 'Pending', 'Delivered'). (Based on observed values)
    # [Uniqueness] Column 'Invoice_Number' MUST BE UNIQUE.
    # [Consistency] Column 'Delivery_Date' MUST BE ON OR AFTER 'Ship_Date'.
    # CRITICAL CONSIDERATIONS:

    # Testable Conditions: The core output must be rules defining clear pass/fail logic for rows.
    # Evidence-Based: Justify every rule with specific profile data. State inferences clearly (e.g., "range inferred from observed min/max", "set inferred from observed unique values").
    # Avoid Descriptive Rules: DO NOT output rules like: "Age is normally distributed", "Status has 3 unique values", "Correlation between X and Y is 0.8", "10% of emails are missing". Convert these observations into actionable validation rules (e.g., "Age MUST BE BETWEEN X AND Y", "Status MUST BE ONE OF ('A','B','C')", "Email MUST NOT BE NULL").
    # Specificity: Rules must reference exact column names from the profile.
    # Row-Level Focus: Ensure rules apply to individual data records.

    # BEGIN ANALYSIS:
    # Now, analyze the provided inputs below and generate the DQ rules following the specified process and format. Focus intensely on translating profile observations into specific, testable, row-level validation conditions.

    # DATASET_PROFILE_CONTEXT:
    # {profile}

    # RETRIEVED_DQ_RULES (Examples for Business Logic Inspiration Only):
    # {retrieved_rules}

    # (LLM Starts Reasoning and Generation Here)
    # """
    # )

    # rule_template = PromptTemplate.from_template(
    # """ 
    # **PERSONA:**

    # You are an exceptionally skilled Senior Data Quality (DQ) Analyst operating within a large insurance enterprise. You possess 
    # deep expertise in data profiling interpretation, DQ frameworks (covering dimensions like Completeness, Uniqueness, Validity, 
    # Accuracy, Consistency, Timeliness), and the practical application of DQ rules to mitigate business risk. You are adept at 
    # translating statistical data profile insights into actionable, specific, and measurable DQ rules. You think critically and 
    # systematically, documenting your reasoning process clearly.

    # **CONTEXT:**

    # We are developing an automated system to suggest relevant Data Quality rules for datasets undergoing onboarding or periodic 
    # review within our insurance operations. We have profiled a specific dataset using `ydata-profiling`, and a filtered summary of 
    # this profile (`DATASET_PROFILE_CONTEXT`) is provided below. Additionally, we have performed a vector similarity search against 
    # our enterprise DQ Rule Library (which contains natural language descriptions of established business logic and DQ checks) based
    # on the dataset's characteristics. The top relevant rules retrieved from this library (`RETRIEVED_DQ_RULES`) are also provided 
    # below. These retrieved rules serve as **guiding principles and examples** of the *types*, *structure*, and *scope* of rules 
    # commonly used in our enterprise framework, but they are **not** specific to the dataset currently being analyzed.

    # **OBJECTIVE:**

    # Your task is to meticulously analyze the provided `DATASET_PROFILE_CONTEXT` and, using the `RETRIEVED_DQ_RULES` as inspiration 
    # and contextual guidance, generate a set of **new, specific, and actionable Data Quality rules** tailored precisely to the 
    # dataset described in the profile. These rules should address potential quality issues identified *directly* from the profile 
    # data.

    # **INPUTS:**

    # 1.  **`DATASET_PROFILE_CONTEXT`**: A structured summary derived from a `ydata-profiling` report for the target dataset. This 
    # context contains key information such as:
    #     *   **Dataset Overview:** Number of variables, observations, missing cells, duplicate rows, data types overview.
    #     *   **Variable Details (Per Column):**
    #         *   Column Name
    #         *   Data Type (Detected, e.g., Numerical, Categorical, Boolean, Date, Text)
    #         *   Basic Statistics (e.g., mean, stddev, min, max, median for numerical; counts, frequencies, unique values for 
    #         categorical/text; date ranges)
    #         *   Missing Value Counts & Percentages
    #         *   Cardinality (Number of distinct values)
    #         *   Presence of Zeros, Negative Values (for numerical)
    #         *   Specific Alerts generated by `ydata-profiling` (e.g., High Correlation, Skewness, High Cardinality, Uniqueness, 
    #             Constant Value, Zeros Percentage, Missing Percentage).
    #     *   **Correlations (if relevant/provided):** Highlighted strong positive or negative correlations between numerical 
    #         variables.
    #     *   **Missing Data Patterns (if relevant/provided):** Matrix or heatmap summaries.

    # 2.  **`RETRIEVED_DQ_RULES`**: A list of natural language DQ rules retrieved from our enterprise library via vector similarity search. These rules represent common patterns and business logic applied elsewhere in the organization. Examples might include:
    #     *   "Policy effective date must be before policy expiration date." (Consistency)
    #     *   "Claim amount must be greater than zero and less than the policy coverage limit." (Validity, Accuracy)
    #     *   "Customer ID must be unique across all active policies." (Uniqueness)
    #     *   "Postal code must conform to the standard 5-digit or 9-digit ZIP code format." (Validity)
    #     *   "Date of birth field should not contain future dates." (Validity)
    #     *   "The 'Status' field must only contain values from the approved list: [Active, Lapsed, Pending, Cancelled]." (Validity - Categorical)
    #     *   "Percentage of missing values in the 'Annual Income' field should not exceed 5%." (Completeness)

    # **INSTRUCTIONS (Chain of Thought & Rule Generation):**

    # Follow these steps meticulously:

    # 1.  **Understand the Goal:** Reiterate the objective: Generate specific DQ rules for the dataset in `DATASET_PROFILE_CONTEXT`, 
    # using `RETRIEVED_DQ_RULES` as guiding examples.
    # 2.  **Deep Dive into `DATASET_PROFILE_CONTEXT`:**
    #     *   Systematically review each section (Overview, Variables, Alerts, etc.).
    #     *   For *each variable/column*, analyze its statistics, data type, missingness, cardinality, and any specific alerts.
    #     *   Identify potential DQ issues or areas needing validation based *solely* on the profile data. Examples:
    #         *   High percentage of missing values in a critical column.
    #         *   Unexpected data types (e.g., numbers in a supposedly text field).
    #         *   Out-of-range values (min/max seem illogical).
    #         *   Very high or very low cardinality suggesting potential issues (e.g., an ID field isn't unique, a status field has 
    #             too many values).
    #         *   Presence of unexpected zeros or negative values.
    #         *   Duplicate rows detected in the overview.
    #         *   Specific alerts flagged by the profiling tool (treat these as high-priority signals).
    #         *   Strong correlations that might imply redundancy or required consistency checks.
    # 3.  **Consult `RETRIEVED_DQ_RULES`:**
    #     *   Review the provided `RETRIEVED_DQ_RULES`.
    #     *   Identify the underlying DQ dimensions they represent (Completeness, Validity, Uniqueness, Consistency, Accuracy, 
    #         Timeliness).
    #     *   Observe the typical structure, level of specificity, and phrasing used in these enterprise rules. Note how they often 
    #         define specific constraints, value sets, formats, or thresholds.
    #     *   Use these rules to *inform* how you structure and phrase your *new* rules, ensuring they align with the enterprise 
    #         style and cover relevant DQ dimensions. **DO NOT simply copy or slightly modify the retrieved rules.** Their purpose is 
    #         guidance, not direct application.
    # 4.  **Reasoning (Chain of Thought - Explicitly Show This):** For each potential DQ issue identified in Step 2, perform the 
    #     following reasoning process:
    #     *   **Observation:** State the specific finding from `DATASET_PROFILE_CONTEXT` (e.g., "Column 'Claim_Amount' has a minimum 
    #         value of -500").
    #     *   **Potential DQ Issue:** Interpret the observation in terms of data quality (e.g., "Negative claim amounts are likely 
    #         invalid and indicate an Accuracy/Validity issue").
    #     *   **Relevant DQ Dimension(s):** Identify the primary DQ dimension(s) involved (e.g., Validity, Accuracy).
    #     *   **Inspiration from Retrieved Rules (if applicable):** Mention if any `RETRIEVED_DQ_RULES` provide a structural pattern 
    #         or address a similar *type* of issue (e.g., "Retrieved rule #2 checks for value ranges, suggesting a rule format like 'Column 
    #         X must be greater than Y'").
    #     *   **Rule Formulation:** Draft a **specific, new, actionable DQ rule** for the target dataset based on the observation and 
    #         reasoning. Make it quantifiable and testable where possible. (e.g., "The 'Claim_Amount' column must contain values greater than 
    #         or equal to 0.").
    #     *   **Justification/Threshold Rationale (Brief):** Briefly explain *why* this rule is needed based on the profile data. If 
    #         setting a threshold (e.g., for missing values), justify it based on the observed percentage or standard practice (if 
    #         inferable, otherwise state the observed value as a point of reference). E.g., "Rule based on observed min value; negative 
    #         claims are illogical." or "Observed missing % is 30%, suggesting a check is needed; proposing a threshold of 10% as a starting 
    #         point, reflecting potential business criticality."
    # 5.  **Synthesize & Refine Rules:**
    #     *   Consolidate the formulated rules.
    #     *   Ensure rules are clear, unambiguous, and directly testable against the data.
    #     *   Prioritize rules addressing alerts or significant deviations noted in the profile.
    #     *   Avoid generating overly generic rules. Focus on what the profile *specifically* indicates.
    #     *   If the profile lacks information to set a precise threshold (e.g., valid category list), formulate the rule to highlight the 
    #         need for such definition (e.g., "The 'Policy_Type' column values should be validated against an approved list of policy types.").
    #     *   Categorize each generated rule by its primary DQ Dimension (e.g., [Completeness], [Validity], [Uniqueness], [Consistency], 
    #         [Accuracy]).

    # **OUTPUT FORMAT:**

    # Present your response in two distinct sections:

    # **Section 1: Chain of Thought Reasoning**
    # *   Provide the detailed step-by-step reasoning process (as outlined in Instruction 4) for *each* generated rule. Make this 
    #     section verbose and clear, showing the connection between the profile data, DQ principles, and the final rule.

    # **Section 2: Generated Data Quality Rules**
    # *   Provide a clean, numbered list of the final generated DQ rules.
    # *   Each rule should be stated clearly and concisely in natural language.
    # *   Prefix each rule with its primary DQ Dimension tag (e.g., `[Validity]`).

    # **CRITICAL CONSIDERATIONS:**

    # *   **Specificity:** Rules must refer to specific column names from the `DATASET_PROFILE_CONTEXT`.
    # *   **Evidence-Based:** Every rule must be justified by specific evidence found within the `DATASET_PROFILE_CONTEXT`. Do not 
    #     invent rules based on assumptions beyond the profile.
    # *   **Actionability:** Rules should describe a condition that can be programmatically checked against the data.
    # *   **No Hallucination:** Do not invent profile details or assume business context not provided. If information is missing to 
    #     make a rule fully concrete (e.g., the exact list of valid categories), formulate the rule to indicate the need for this 
    #     external information.
    # *   **Leverage Alerts:** Pay special attention to any explicit alerts mentioned in the `DATASET_PROFILE_CONTEXT`.

    # **BEGIN ANALYSIS:**

    # Now, analyze the provided inputs below and generate the DQ rules following the specified process and format.
    # {profile}

    # Additional Context for the Data:
    # {context}


    # **`RETRIEVED_DQ_RULES`:**
    # {retrieved_rules}

    # ---

    # **(LLM Starts Reasoning and Generation Here)**
    # """
    # )
 
    rule_template = PromptTemplate.from_template(
    """ 
    PERSONA:
 
    You are an exceptionally skilled Senior Data Quality (DQ) Analyst AI, 
    expert in interpreting data profiles (ydata-profiling), understanding DQ dimensions (covering dimensions like Uniqueness, Validity,
    Accuracy, Consistency, Timeliness, Completeness, etc.), 
    and translating statistical findings into precise, actionable, 
    row-level validation rules suitable for automated checks in SQL or Python/pandas. 
    You prioritize clarity, specificity, and evidence-based reasoning.  
    You are also proficient in SQL and Python/pandas, enabling you to implement these rules effectively.
    
    CONTEXT:
    
    We're automating DQ rule suggestions. You are given a summary of a dataset's profile (DATASET_PROFILE_CONTEXT) and a list of general DQ rules (RETRIEVED_DQ_RULES) retrieved from our enterprise library via vector search. These retrieved rules are examples only, illustrating the types of validation logic (range checks, format validation, set membership, null checks, uniqueness, consistency) and the expected structure/phrasing for rules within our framework. They DO NOT contain logic specific to the current dataset.
    
    OBJECTIVE:
    
    Your primary goal is to generate a set of Data Quality rules defining specific, testable pass/fail conditions for individual rows of the dataset described in DATASET_PROFILE_CONTEXT. These rules must be directly derived from the profile's statistics, observed values, and alerts. They should be immediately usable for implementing automated data validation scripts (e.g., in SQL WHERE clauses or pandas boolean masks), moving significantly beyond just summarizing profile alerts. Use the RETRIEVED_DQ_RULES solely for understanding how to structure validation logic and phrase the final rules clearly.
    
    INPUTS:
    
    DATASET PROFILE - Structured summary from ydata-profiling:
    {profile}

    RETRIEVED_DQ_RULES - for insights on business logic (non-standard data quality checks):
    {retrieved_rules}

    Optionally, ADDITIONAL CONTEXT (data dictionary, business guidelines, etc.):
    {context}
    
    INSTRUCTIONS (Chain of Thought & Rule Generation):
    
    Follow these steps meticulously:
    
    Understand Goal: Generate specific, row-level validation rules (pass/fail conditions) based on the profile, inspired by retrieved rules.
    Analyze DATASET_PROFILE_CONTEXT: Systematically review overview, variables, and alerts. For each column, identify potential needs for validation based on its type, stats, observed values (min/max/unique counts), and alerts. Focus on translating observations into potential checks.
    Use RETRIEVED_DQ_RULES for Structural Insight: Review retrieved rules to understand how checks for ranges, formats, sets, nullity, uniqueness, or consistency are typically expressed. Apply these patterns to the specific findings from the profile. DO NOT: Copy rules, slightly modify rules, assume business logic from retrieved rules, or use them if they don't match a type of validation suggested by the profile.
    Reasoning (Chain of Thought - Explicitly Show This): For each validation need identified in Step 2:
    Observation: State the finding from DATASET_PROFILE_CONTEXT (e.g., "Column 'Age' (Numeric): min=0, max=115, 0.5% missing.").
    Potential Validation Need & Type: Interpret the need for a specific check (e.g., "Need to ensure Age is not missing and stays within the observed, reasonable range. Requires a Non-Null Check and a Range Check.").
    Relevant DQ Dimension(s): (e.g., Completeness, Validity).
    Inspiration: Retrieved rules may contain relevant business logic pertaining to a sector (manufacturing, banking, insurance, etc).
    Rule Formulation (Define Testable Condition): Draft a specific DQ rule defining a clear pass/fail condition for a row. Use imperative language. Infer reasonable constraints from profile data where appropriate, clearly stating the basis for inference.
    Standard Sanity Checks that should be present (based on YData Profile):
    For identifiers: Propose nullity checks and consistency checks for potential identifier fields (for e.g., differing Customer Names for the same Customer ID should be flagged). Do not propose ranges and bounds.
    For Text: If format is suggested (e.g., always 5 digits, looks like email), propose MUST MATCH REGEX '...'. If uniqueness alert exists, propose MUST BE UNIQUE.
    For Dates: Propose MUST BE a valid date [in YYYY-MM-DD format] or MUST NOT BE NULL or MUST BE ON OR BEFORE {{current_date}} (if future dates observed).
    For Missing % Alert: Propose MUST NOT BE NULL or MUST NOT BE empty.
    For Duplicate Rows: There must be no duplicate rows.
    For special fields: Propose inferred format validity patterns for special fields such as PAN, GST Number, etc

    Justification: Briefly explain why the rule/condition is needed based on the profile observation/inference (e.g., "Ensures age validity based on observed range", "Restricts status to observed values, requires business confirmation", "Addresses non-unique ID alert").
    Synthesize & Refine Rules:
    Consolidate rules. Ensure they define testable, row-level conditions.
    Prioritize rules that directly translate profile stats/alerts into checks (Nullity, Range, Format, Set Membership, Uniqueness, Consistency) over purely descriptive statistical observations.
    Ensure clear, imperative phrasing (e.g., "MUST BE", "MUST NOT BE NULL", "MUST MATCH REGEX", "MUST BE ONE OF").
    Tag each rule with its primary DQ Dimension.

    OUTPUT FORMAT:
    
    Present your response in two distinct sections:
    
    Section 1: Chain of Thought Reasoning
    
    Provide the detailed step-by-step reasoning process (Instruction 4) for each generated rule, clearly showing the link from profile observation to the specific validation condition formulation.
    Section 2: Generated Data Quality Rules (Row-Level Validation Conditions)
    
    Provide a clean, numbered list of the final generated DQ rules.
    Each rule must clearly state a testable condition using imperative language.
    Prefix each rule with its primary DQ Dimension tag (e.g., [Validity]).
    Example Rule Formulations:
    [Completeness] Column 'Order_ID' MUST NOT BE NULL.
    [Validity] Column 'Order_Amount' MUST BE GREATER THAN OR EQUAL TO 0.
    [Validity] Column 'Product_Code' MUST MATCH REGEX '^[A-Z]{{3}}-[0-9]{{5}}$'. (Based on inferred pattern)
    [Validity] Column 'Ship_Status' MUST BE ONE OF ('Shipped', 'Pending', 'Delivered'). (Based on observed and logical values)
    [Uniqueness] Column 'Invoice_Number' MUST BE UNIQUE.
    [Consistency] Column 'Delivery_Date' MUST BE ON OR AFTER 'Ship_Date'.
    [Deduplication] There should be no duplicate rows.

    CRITICAL CONSIDERATIONS:
    
    Testable Conditions: The core output must be rules defining clear pass/fail logic for rows.
    Evidence-Based: Justify every rule with specific profile data. State inferences clearly (e.g., "range inferred from observed min/max", "set inferred from observed unique values").
    Avoid Descriptive Rules: DO NOT output rules like: "Age is normally distributed", "Status has 3 unique values", "Correlation between X and Y is 0.8", "10% of emails are missing". Convert these observations into actionable validation rules (e.g., "Age MUST BE BETWEEN X AND Y", "Status MUST BE ONE OF ('A','B','C')", "Email MUST NOT BE NULL").
    Specificity: Rules must reference exact column names from the profile.
    Row-Level Focus: Ensure rules apply to individual data records.
    Identifiers: Only apply nullity and uniqueness checks to potential identifier fields. Do not apply ranges to potential identifier fields.

    BEGIN ANALYSIS:
    
    Now, analyze the provided inputs and generate the DQ rules following the specified process and format. Focus intensely on translating profile observations into specific, testable, row-level validation conditions.
    Finally, end your response with a "------".

    (LLM Starts Reasoning and Generation Here)
    """
    )

    continue_rule_template = PromptTemplate.from_template(
    """
    You are resuming a task where you were acting as an exceptionally skilled Senior Data Quality (DQ) Analyst AI.
    Original Objective Recap: Your goal was to generate specific, actionable, row-level Data Quality rules defining testable pass/fail conditions based on a provided dataset profile (DATASET_PROFILE_CONTEXT), 
    using a list of retrieved general rules (RETRIEVED_DQ_RULES) for business context and inspiration only. 
    The output needed to be in two sections: detailed Chain of Thought (CoT) reasoning, followed by a numbered list of specific DQ rules tagged with their dimension.
    
    ORIGINAL INPUTS:
    
    DATASET PROFILE - Structured summary from ydata-profiling:
    {profile}

    RETRIEVED_DQ_RULES - for insights on business logic (non-standard data quality checks):
    {retrieved_rules}

    Optionally, ADDITIONAL CONTEXT (data dictionary, business guidelines, etc.):
    {context}

    ORIGINAL INSTRUCTIONS Recap (Key Points to Continue Following):
    Follow these steps meticulously:
    
    Understand Goal: Generate specific, row-level validation rules (pass/fail conditions) based on the profile, inspired by retrieved rules.
    Analyze DATASET PROFILE: Systematically review overview, variables, and alerts. For each column, identify potential needs for validation based on its type, stats, observed values (min/max/unique counts), and alerts. Focus on translating observations into potential checks.
    Use RETRIEVED_DQ_RULES for Structural Insight: Review retrieved rules to understand how checks for ranges, formats, sets, nullity, uniqueness, or consistency are typically expressed. Apply these patterns to the specific findings from the profile. DO NOT: Copy rules, slightly modify rules, assume business logic from retrieved rules, or use them if they don't match a type of validation suggested by the profile.
    Reasoning (Chain of Thought - Explicitly Show This): For each validation need identified in Step 2:
    Observation: State the finding from DATASET PROFILE (e.g., "Column 'Age' (Numeric): min=0, max=115, 0.5% missing.").
    Potential Validation Need & Type: Interpret the need for a specific check (e.g., "Need to ensure Age is not missing and stays within the observed, reasonable range. Requires a Non-Null Check and a Range Check.").
    Relevant DQ Dimension(s): (e.g., Completeness, Validity).
    Inspiration: Retrieved rules may contain relevant business logic pertaining to a sector (manufacturing, banking, insurance, etc).
    Rule Formulation (Define Testable Condition): Draft a specific DQ rule defining a clear pass/fail condition for a row. Use imperative language. Infer reasonable constraints from profile data where appropriate, clearly stating the basis for inference.
    Standard Sanity Checks that should be present (based on YData Profile):
    For identifiers: Propose nullity checks and consistency checks for potential identifier fields (for e.g., differing Customer Names for the same Customer ID should be flagged). Do not propose ranges and bounds.
    For Text: If format is suggested (e.g., always 5 digits, looks like email), propose MUST MATCH REGEX '...'. If uniqueness alert exists, propose MUST BE UNIQUE.
    For Dates: Propose MUST BE a valid date [in YYYY-MM-DD format] or MUST NOT BE NULL or MUST BE ON OR BEFORE {{current_date}} (if future dates observed).
    For Missing % Alert: Propose MUST NOT BE NULL or MUST NOT BE empty.
    For Duplicate Rows: There must be no duplicate rows.
    For special fields: Propose inferred format validity patterns for special fields such as PAN, GST Number, etc

    Justification: Briefly explain why the rule/condition is needed based on the profile observation/inference (e.g., "Ensures age validity based on observed range", "Restricts status to observed values, requires business confirmation", "Addresses non-unique ID alert").
    Synthesize & Refine Rules:
    Consolidate rules. Ensure they define testable, row-level conditions.
    Prioritize rules that directly translate profile stats/alerts into checks (Nullity, Range, Format, Set Membership, Uniqueness, Consistency) over purely descriptive statistical observations.
    Ensure clear, imperative phrasing (e.g., "MUST BE", "MUST NOT BE NULL", "MUST MATCH REGEX", "MUST BE ONE OF").
    Tag each rule with its primary DQ Dimension.

    OUTPUT FORMAT:
    
    Present your response in two distinct sections:
    
    Section 1: Chain of Thought Reasoning
    
    Provide the detailed step-by-step reasoning process (Instruction 4) for each generated rule, clearly showing the link from profile observation to the specific validation condition formulation.
    Section 2: Generated Data Quality Rules (Row-Level Validation Conditions)
    
    Provide a clean, numbered list of the final generated DQ rules.
    Each rule must clearly state a testable condition using imperative language.
    Prefix each rule with its primary DQ Dimension tag (e.g., [Validity]).
    Example Rule Formulations:
    [Completeness] Column 'Order_ID' MUST NOT BE NULL.
    [Validity] Column 'Order_Amount' MUST BE GREATER THAN OR EQUAL TO 0.
    [Validity] Column 'Product_Code' MUST MATCH REGEX '^[A-Z]{{3}}-[0-9]{{5}}$'. (Based on inferred pattern)
    [Validity] Column 'Ship_Status' MUST BE ONE OF ('Shipped', 'Pending', 'Delivered'). (Based on observed and logical values)
    [Uniqueness] Column 'Invoice_Number' MUST BE UNIQUE.
    [Consistency] Column 'Delivery_Date' MUST BE ON OR AFTER 'Ship_Date'.
    [Deduplication] There should be no duplicate rows

    CRITICAL CONSIDERATIONS:
    
    Testable Conditions: The core output must be rules defining clear pass/fail logic for rows.
    Evidence-Based: Justify every rule with specific profile data. State inferences clearly (e.g., "range inferred from observed min/max", "set inferred from observed unique values").
    Avoid Descriptive Rules: DO NOT output rules like: "Age is normally distributed", "Status has 3 unique values", "Correlation between X and Y is 0.8", "10% of emails are missing". Convert these observations into actionable validation rules (e.g., "Age MUST BE BETWEEN X AND Y", "Status MUST BE ONE OF ('A','B','C')", "Email MUST NOT BE NULL").
    Specificity: Rules must reference exact column names from the profile.
    Row-Level Focus: Ensure rules apply to individual data records.
    Identifiers: Only apply nullity and uniqueness checks to potential identifier fields. Do not apply ranges to potential identifier fields.
    
    PROGRESS SO FAR (Previous Output - Cut Off): This is the complete text you generated in the previous turn before stopping:
    {output}
    
    Your Task Now: Continue generating the response exactly where it left off in the text provided above under "Progress So Far". 
    Do not repeat any content from that section. Follow all the original instructions precisely, maintaining the Chain of Thought reasoning process and the 
    specified output format (Section 1: CoT, Section 2: Rules list). Ensure you complete the analysis for all relevant findings in the original DATASET_PROFILE_CONTEXT 
    that were not yet covered, generating the corresponding rules and their CoT. Start your response immediately with the next part of 
    the content, without any preamble like "Okay, continuing where I left off...". End your response with a "------". 

    (LLM Continues Generation Here) 
    """
    )

    # rule_template = PromptTemplate.from_template(
    # """
    # You are an expert data analyst who can create data quality rules for a dataset by analysing and drawing inspiration from similar data quality rules.
    # Following is a ydata profile of the dataset:\n{profile}\n\n
    # Analyze the following list of data quality rules alongside the kind of field they are for:\n{retrieved_rules}\n\n
    # Following is some additional context for the data (for e.g., data dictionary): {context}
    # Some of the given data quality rules may not directly pertain to fields present in the given dataset. 
    # Assess if the fields in the given dataset bear any relevance to the kinds of fields mentioned in the data quality rules I provided 
    # based on the context from the dataset. Infer similar data quality rules so that they can be used for validating data in the dataset.
    # If a common rule applies to multiple columns, it should be listed as a single list item (for e.g., column_1, column_2 and column_3 must not be null)
    # If a data quality rule from the set of data quality rules I provided is not applicable to the dataset at all, DO NOT use it to infer relevant data quality rules.
    # Respond in the following format:
    # Data Quality Rules:
    # [List of data quality rules]\n
    # Generate a list of data quality rules and end your response with "------".
    # Respond with the data quality rules only. Do not include column descriptions, introductions and conclusions.
    # """
    # )

    # continue_rule_template = PromptTemplate.from_template(
    # """
    # You are an expert data analyst who can create data quality rules for a dataset by analysing and drawing inspiration from similar data quality rules.
    # Following is a ydata profile of the dataset:\n{profile}\n\n
    # Analyze the following list of data quality rules alongside the kind of field they are for:\n{retrieved_rules}\n\n
    # Following is some additional context for the data (for e.g., data dictionary): {context}
    # Some of the given data quality rules may not directly pertain to fields present in the given dataset. 
    # Assess if the fields in the given dataset bear any relevance to the kinds of fields mentioned in the data quality rules I provided 
    # based on the profile and additional context from the dataset. Infer similar data quality rules so that they can be used for validating data in the dataset.
    # If a common rule applies to multiple columns, it should be listed as a single list item (for e.g., column_1, column_2 and column_3 must not be null)
    # If a data quality rule from the set of data quality rules I provided is not applicable to the dataset at all, DO NOT use it to infer relevant data quality rules.
    # Respond in the following format:
    # Data Quality Rules:
    # [List of data quality rules]\n

    # Following are the rules you generated so far:
    # {output}

    # Generate the remaining data quality rules and end your response with "------".
    # Respond with the data quality rules only. Do not include column descriptions, introductions and conclusions.
    # """
    # )

    vector_query_chain = runnable | generate_vector_query_template | llm | output_parser
    queries_str = vector_query_chain.invoke({"profile": profile, "context": context})
    queries = queries_str.split(", ")
 
    for query in queries:
        rules_with_scores = vector_store.similarity_search_with_relevance_scores(query, k=3, kwargs={"score_threshold":0.8})
        rules = [rule for rule, _ in rules_with_scores]
        retrieved_rules.extend([doc.page_content for doc in rules])
    
    # st.info(retrieved_rules)

    rule_chain = runnable | rule_template | llm | output_parser
    continue_rule_chain = runnable | continue_rule_template | llm | output_parser
    new_rules = rule_chain.invoke({"profile": profile, "context": context, "retrieved_rules": str(retrieved_rules)})

    while not "------" in new_rules:
        new_rules = "\n".join(new_rules.split("\n")[:-1])
        continuation = continue_rule_chain.invoke({"output": new_rules, "context": context, "retrieved_rules": str(retrieved_rules), "profile": profile})
        new_rules += "\n" + continuation
    
    new_rules = new_rules[new_rules.find("Generated Data Quality Rules"):]
 
    return new_rules

def normalise_rules(rules, glossary):
    normalise_template = PromptTemplate.from_template(
    """ 
    I will provide you with a set of data quality rules for a table and a data dictionary of columns in the table. Your task
    is to rewrite these rules by referring to the columns with easily understandable names. Your changes should not affect any
    other part of the given rules except for the column names. 

    Example:
    Not satisfactory:
    [Completeness] Column 'Postal date' MUST NOT BE NULL.

    Satisfactory:
    [Completeness] Postal date MUST NOT BE NULL.

    Glossary:
    {glossary}

    Data Quality Rules:
    {rules}

    Respond with the rewritten rules only. Do not include introductions, explanations and conclusions. Do not repeat rules.
    """
    )

    normalise_chain = runnable | normalise_template | llm | output_parser
    normalised_rules = normalise_chain.invoke({"glossary": glossary, "rules": rules})

    return normalised_rules

def create_sql(rules, ddl, dialect):
    sql_template = PromptTemplate.from_template(
    """
    You are an expert SQL developer specializing in data validation and quality checks.
    Your task is to generate SQL code based on a set of Data Quality (DQ) rules provided.

    The goal is to create SQL script in the {dialect} dialect that, when executed, will:
    1.  Create a new table such that its name is prefixed with the schema name, begins with validated_ followed by the table name, with the exact same schema as the source table.
    2.  Create a new table such that its name is prefixed with the schema name, begins with quarantined_ followed by the table name, with the exact same schema as the source table  PLUS an additional column 
    named `QuarantineReason` (TEXT or VARCHAR type) to briefly state why the row failed validation.
    3.  Populate the validated table with all rows from the source table that satisfy ALL the provided DQ rules.
    4.  Populate the quarantined table with all rows from source table that fail AT LEAST ONE of the provided DQ rules. Include a 
    brief reason in the `QuarantineReason` column (e.g., 'Age out of range', 'Invalid Email Format', 'Missing Status').

    Context:
    * **Schema DDL (Structure):**
        ```sql
        {table_ddl}
        ```
    * **Data Quality Rules to Implement:**
        ```text
        {rules}
        ```

    **Instructions for SQL Generation:**
    * Generate SQL compatible with the mentioned dialect.
    * Consider the possibility of validated and quarantined tables already existing.
    * First, include the `CREATE TABLE AS` or `CREATE TABLE LIKE` statement for the validated table, depending on the dialect. Ensure the WHERE 1=0 clause is added so that the table is initially empty.
    * Second, include the `CREATE TABLE AS` or `CREATE TABLE LIKE` statement for the quarantined table depending on the dialect. Ensure the WHERE 1=0 clause is added so that the table is initially empty. Add an `ALTER TABLE` statement for the `QuarantineReason` column.
    * Third, write the INSERT SELECT statement for the validated table with a `WHERE` clause combining ALL DQ rules with AND 
    conditions. Select all columns from the source table.
    * Fourth, write the INSERT SELECT statement for the quarantined table. Select all columns from the source table PLUS 
    construct the `QuarantineReason` string. The `WHERE` clause should select rows that DO NOT meet the combined AND conditions 
    used for the validated table (i.e., fail one or more rules). Use CASE statements within the SELECT clause or other methods 
    to determine the `QuarantineReason`.
    * Ensure column names are correctly referenced based on the provided DDL. Pay attention to data types when writing conditions (e.g., comparing numbers, strings, dates).
    * **Output ONLY the complete SQL script.** Do not include any explanations, greetings, or markdown formatting around the SQL 
    code block.

    **Example DQ Rule Translation:**
    * If a rule is "Age must be >= 0 AND Age <= 100", the condition in the validated table INSERT would include `(Age >= 0 AND 
    Age <= 100)`. The quarantined table INSERT would select rows `WHERE NOT (Age >= 0 AND Age <= 100)` and set `QuarantineReason` 
    potentially using `CASE WHEN Age < 0 THEN 'Age below minimum' WHEN Age > 100 THEN 'Age exceeds maximum' ELSE 'Other reason' 
    END`.

    Generate the SQL code now and end with "------":
    """
    ) 

    continue_sql_template = PromptTemplate.from_template(
    """ 
    You are an expert SQL developer specializing in data validation and quality checks.
    Your task was to generate SQL code based on a set of Data Quality (DQ) rules provided.

    You followed these instructions to generate part of the code:
    The goal is to create SQL script in the {dialect} dialect that, when executed, will:
    1.  Create a new table such that its name is prefixed with the schema name, begins with validated_ followed by the table name, with the exact same schema as the source table.
    2.  Create a new table such that its name is prefixed with the schema name, begins with quarantined_ followed by the table name, with the exact same schema as the source table  PLUS an additional column 
    named `QuarantineReason` (TEXT or VARCHAR type) to briefly state why the row failed validation.
    3.  Populate the validated table with all rows from the source table that satisfy ALL the provided DQ rules.
    4.  Populate the quarantined table with all rows from source table that fail AT LEAST ONE of the provided DQ rules. Include a 
    brief reason in the `QuarantineReason` column (e.g., 'Age out of range', 'Invalid Email Format', 'Missing Status').

    Context:
    * **Schema DDL (Structure):**
        ```sql
        {table_ddl}
        ```
    * **Data Quality Rules to Implement:**
        ```text
        {rules}
        ```

    **Instructions for SQL Generation:**
    * Generate SQL compatible with the mentioned dialect.
    * Consider the possibility of validated and quarantined tables already existing.
    * First, include the `CREATE TABLE AS` or `CREATE TABLE LIKE` statement for the validated table, depending on the dialect. Ensure the WHERE 1=0 clause is added so that the table is initially empty.
    * Second, include the `CREATE TABLE AS` or `CREATE TABLE LIKE` statement for the quarantined table depending on the dialect. Ensure the WHERE 1=0 clause is added so that the table is initially empty. Add an `ALTER TABLE` statement for the `QuarantineReason` column.
    * Third, write the INSERT SELECT statement for the validated table with a `WHERE` clause combining ALL DQ rules with AND 
    conditions. Select all columns from the source table.
    * Fourth, write the INSERT SELECT statement for the quarantined table. Select all columns from the source table PLUS 
    construct the `QuarantineReason` string. The `WHERE` clause should select rows that DO NOT meet the combined AND conditions 
    used for the validated table (i.e., fail one or more rules). Use CASE statements within the SELECT clause or other methods 
    to determine the `QuarantineReason`.
    * Ensure column names are correctly referenced based on the provided DDL. Pay attention to data types when writing conditions (e.g., comparing numbers, strings, dates).
    * **Output ONLY the complete SQL script.** Do not include any explanations, greetings, or markdown formatting around the SQL 
    code block.

    **Example DQ Rule Translation:**
    * If a rule is "Age must be >= 0 AND Age <= 100", the condition in the validated table INSERT would include `(Age >= 0 AND 
    Age <= 100)`. The quarantined table INSERT would select rows `WHERE NOT (Age >= 0 AND Age <= 100)` and set `QuarantineReason` 
    potentially using `CASE WHEN Age < 0 THEN 'Age below minimum' WHEN Age > 100 THEN 'Age exceeds maximum' ELSE 'Other reason' 
    END`.

    Following is the code you have generated so far:
    {code}

    Generate the remaining SQL code now and end with "------":
    """
    )

    table_ddl_str = "\n\n\n".join([f"Table: {key}\n\n{value}" for key, value in ddl.items()])
    rules_str = "\n\n\n".join([f"Table: {key}\n\n{value}" for key, value in rules.items()])

    sql_chain = runnable | sql_template | llm | output_parser
    continue_sql_chain = runnable | continue_sql_template | llm | output_parser

    code = sql_chain.invoke({"table_ddl": table_ddl_str, "rules": rules_str, "dialect": dialect})
    code = code.replace("```sql", "").replace("```", "")

    while not code.endswith("------"):
        if not code.endswith(");"):
            code = "\n".join(code.split("\n")[:-1])
        continuation = continue_sql_chain.invoke({"table_ddl": table_ddl_str, "rules": rules_str, "dialect": dialect, "code": code})
        code += "\n" + continuation
        code = code.replace("```sql", "").replace("```", "")
    
    code = code.replace("------", "")

    return code

def create_pandas(rules, info):
    pandas_template = PromptTemplate.from_template(
    """ 
    You are an expert Python developer specializing in data manipulation and validation using the pandas library.
    Your task is to generate Python code based on a set of Data Quality (DQ) rules provided.
    
    The goal is to create a Python script that, when executed, will:
    1.  Take an existing pandas DataFrame (named `df`) as input.
    2.  Create a new pandas DataFrame named `validated_df` containing only the rows from `df` that satisfy ALL the provided DQ rules.
    3.  Create another new pandas DataFrame named `quarantined_df` containing only the rows from `df` that fail AT LEAST ONE of the 
    provided DQ rules. This DataFrame should include all original columns PLUS an additional column named `QuarantineReason` 
    (string type) to briefly state the first DQ rule that the row failed.
    
    **Context:**
    * **DataFrame Information:** Assume the input DataFrame is in a variable named `df`.

    * **Columns and Types with Sample of rows:**
        {info}

    * **Data Quality Rules to Implement:**
        ```text
        {rules}
        ```
    
    **Instructions for Python/Pandas Code Generation:**
    * Write import statements for the required libraries.
    * Generate clean, efficient Python code using the pandas library.
    * The code should define two DataFrames: `validated_df` and `quarantined_df`.
    * Use boolean indexing on `df` to select rows for each output DataFrame.
    * To create `validated_df`, combine ALL individual DQ rule conditions using logical AND (`&`).
    * To create `quarantined_df`, select rows that DO NOT meet the combined conditions for the validated DataFrame (i.e., use the inverse of the combined boolean mask).
    * For `quarantined_df`, implement logic to populate the `QuarantineReason` column. This should indicate the *first* rule (in the order provided or a logical checking order) that a row failed. Suggest using methods like boolean masks for each rule, `np.select`, or `.apply()` with a helper function to determine the reason.
    * Pay close attention to data types when writing conditions. Handle potential errors during comparisons or type conversions (e.g., use `pd.to_numeric(errors='coerce')`, `pd.to_datetime(errors='coerce')`). Handle `NaN`/`None` values appropriately within the conditions based on the rule's intent (e.g., using `.isnull()`, `.notnull()`, `.fillna()`).
    * Ensure column names are written correctly in the code based on the provided column information.
    * Ignore index.
    * If a rule is formatted like "'<Rule Name>': <Details of rules>", the `QuarantineReason` should be the same as the Rule Name.
    * Not every DQ Rule directly may directly reference a particular column. Some rules may instruct checks across multiple columns (for e.g., standardizing unit of measures). You must intelligently infer the appropriate column for applying the DQ Rule by referring to the source dataframe information.
    * **Output ONLY the complete Python code block.** Do not include import statements, explanations, greetings, or markdown formatting around the code block. Assume `df` exists.
    
    **Example DQ Rule Translation (Pandas):**
    * If a rule is "'Age out of range': Age must be >= 0 AND Age <= 100":
        * The boolean condition for validation would involve something like: `(pd.to_numeric(df['age'], errors='coerce').notnull()) & (df['age'] >= 0) & (df['age'] <= 100)` (adjust NaN handling as needed).
        * For the `QuarantineReason` in `quarantined_df`, the logic might look like:
            ```python
            # Example using np.select (inside the generated code)
            conditions = [
                df['age'].isnull(),
                pd.to_numeric(df['age'], errors='coerce').isnull(), # Failed conversion
                df['age'] < 0 and df['age'] > 100,
                df['age'] > 100
            ]
            choices = [
                'Age is missing',
                'Age is not numeric',
                'Age out of range'
            ]
            # Apply this logic specifically to rows that failed the overall validation
            # This is often done after identifying the failing rows.
            # A full implementation might apply this within a function or chained .loc
            ```
    
    Generate the Python code using pandas now:
    """
    )

    pandas_chain = runnable | pandas_template | llm | output_parser
    code = pandas_chain.invoke({"info": info, "rules": rules})

    code = code.replace("```python", "").replace("```", "")

    return code

def fix_pandas(code, error):
   fix_pandas_prompt = PromptTemplate.from_template(
   """ 
   Following is a pandas code to manipulate a dataframe called 'df':
   {code}

   Following is the error message received upon executing it:
   {error}

   Fix the code so that it is syntactically correct. Respond with the fixed code only. Do not include introductions, explanations and 
   conclusions.
   """
   )

   fix_pandas_chain = runnable | fix_pandas_prompt | llm | output_parser
   code = fix_pandas_chain.invoke({"code": code, "error": error})

   code = code.replace("```python", "").replace("```", "")

   return code


 
# Streamlit Interface
 
# st.title("YData Profiling Report Viewer")
# st.write("Upload your JSON report generated by ydata-profiling to view key sections.")

# engine = None
# glossary = None

if 'engine' not in st.session_state:
    st.session_state.engine = None
if 'checkbox_states' not in st.session_state:
    st.session_state.checkbox_states = {}
if 'loaded_dataframes' not in st.session_state:
    st.session_state.loaded_dataframes = {}
if 'profile_data' not in st.session_state:
    st.session_state.profile_data = {}
if 'rules' not in st.session_state:
    st.session_state.rules = {}
if 'uploaded_rules' not in st.session_state:
    st.session_state.uploaded_rules = None
if 'normalised_rules' not in st.session_state:
    st.session_state.normalised_rules = {}
if 'table_ddl' not in st.session_state:
    st.session_state.table_ddl = {}
if 'complete_glossary' not in st.session_state:
    st.session_state.complete_glossary = {}
if 'glossary' not in st.session_state:
    st.session_state.glossary = {}
if "code" not in st.session_state:
    st.session_state.code = None
if 'cleansed_data' not in st.session_state:
    st.session_state.cleansed_data = {}

 
st.set_page_config(
    page_title="Data Quality Management with AI",
    page_icon="✅",
    layout="wide"
)
st.title("Manage Data Quality with AI ✅")
 
with st.sidebar:
   
    # db_type = st.sidebar.selectbox("Database Type", ["None", "MySQL", "PostgreSQL", "MS SQL Server"])
 
    with st.expander("Create Connections"):
        db_type_expander = st.selectbox("Database Type", ["PostgreSQL", "MySQL", "Microsoft SQL Server", "OracleDB", "SQLite"], key="db_type_expander")
 
        connection_details = {}
        if db_type_expander == "SQLite":
            connection_details['database'] = st.text_input("Database File Path", key="sqlite_db_path")
        else:
            connection_details['host'] = st.text_input("Host", type="password", key="db_host")
            connection_details['username'] = st.text_input("Username", type="password", key="db_user")
            connection_details['password'] = st.text_input("Password", type="password", key="db_pass")
            if db_type_expander != "OracleDB":
                connection_details['database'] = st.text_input("Database Name", type="password", key="db_name")
            if db_type_expander in ["PostgreSQL", "MySQL"]:
                connection_details['port'] = st.text_input("Port", key="db_port")
            if db_type_expander == "OracleDB":
                connection_details['service'] = st.text_input("Service Name", key="db_service")
            if db_type_expander == "Microsoft SQL Server":
                connection_details['authentication'] = st.selectbox("Authentication Type", ["Windows Authentication", "Others"], key="db_auth")
 
        glossary = st.file_uploader("Upload Glossary", type=["xlsx"])

        if st.button("Add Data Source"):
            config = {
                'rdbms': db_type_expander,
                **connection_details
            }
 
            with st.spinner("Testing connection..."):
                engine, error = create_db_engine(config)
                if engine:
                    st.session_state.engine = engine
                if error:
                    st.error(error)
                else:
                    st.success("Connection successful!")
 
    st.write("### OR")
 
    uploaded_file = st.file_uploader(
        "Upload a Dataset",
        type=["csv", "xlsx"]
    )

    uploaded_rules = st.text_area("Provide DQ Rules (Optional)")
    if uploaded_rules.strip() != "":
        st.session_state.rules['dataset'] = uploaded_rules
    if uploaded_rules.strip() == "" and 'dataset' in st.session_state.rules.keys():
        del st.session_state.rules['dataset']
 
if st.session_state.engine is not None:
    # Get table names
    with st.spinner("Fetching Tables..."):
        table_names = get_table_names(st.session_state.engine)

    if glossary:
        st.session_state.complete_glossary = get_business_glossary(glossary)
    
    if not table_names:
        st.warning("No tables found in the database or failed to fetch them.")
    else:
        st.subheader("Select Tables to Load")
 
        # Store current checkbox states
        st.session_state.checkbox_states = {}
        cols = st.columns(4) # Adjust number of columns
        col_index = 0
        for table in sorted(table_names):
            with cols[col_index % len(cols)]:
                # Use the table name as the key for the checkbox state
                st.session_state.checkbox_states[table] = st.checkbox(
                    table,
                    key=f"cb_{table}",
                    # Set default value based on whether it's already loaded
                    value=(table in st.session_state.loaded_dataframes)
                 )
            col_index += 1
 
        # Logic to load/unload based on checkbox changes
        for table_name, is_selected in st.session_state.checkbox_states.items():
            is_loaded = table_name in st.session_state.loaded_dataframes
 
            # Case 1: Selected but not loaded -> Load it
            if is_selected and not is_loaded:
                with st.spinner(f"Loading full table `{table_name}`..."):
                    # df_full = fetch_full_table(st.session_state.engine, table_name)
                    df_full = fetch_table_data(st.session_state.engine, table_name, n=20)

                    schema, table = table_name.split(".")
                    metadata = MetaData()
                    metadata.reflect(bind=st.session_state.engine, schema=schema)
                    table_obj = metadata.tables.get(table_name)
                    # st.info(metadata.tables.keys())
                    if table_obj is not None:
                        table_ddl = str(CreateTable(table_obj).compile(st.session_state.engine))
                        st.session_state.table_ddl[table_name] = table_ddl

                    if not df_full.empty:
                        st.session_state.loaded_dataframes[table_name] = df_full
                        
                        # st.session_state.glossary[table_name]['description'] = table_description[table_name]
                        # st.session_state.glossary[table_name]['column_descriptions'] = column_description[table_name]
                        # st.success(f"Table `{table_name}` loaded successfully ({len(df_full)} rows).") 
                    else:
                         # Error handled/displayed in fetch function
                         st.error(f"Failed to load `{table_name}`.")
 
            # Case 2: Not selected but currently loaded -> Unload it
            elif not is_selected and is_loaded:
                del st.session_state.loaded_dataframes[table_name]
                del st.session_state.table_ddl[table_name]
                del st.session_state.glossary[table_name]

            selected_glossary = {table: st.session_state.complete_glossary[table] for table in st.session_state.checkbox_states if st.session_state.checkbox_states[table]}
            st.session_state.glossary = selected_glossary

# profile_data = None # Initialize profile_data outside the conditional block
 
# if (uploaded_file is not None or st.session_state.loaded_dataframes != {}) and not os.path.exists(PROFILE_JSON_PATH):
if (uploaded_file is not None or st.session_state.loaded_dataframes != {}):
    # st.json(st.session_state.table_ddl)
    if uploaded_file is not None:
        try:
            file_extension = uploaded_file.name.split('.')[-1].lower()
            if file_extension == 'csv':
                df = pd.read_csv(uploaded_file)
            elif file_extension == 'xlsx':
                df = pd.read_excel(uploaded_file)
            else:
                st.error("Unsupported file type. Please upload a CSV or XLSX file.")
                df = None

            if df is not None:
                st.session_state.loaded_dataframes['dataset'] = df
                if 'dataset' not in st.session_state.profile_data.keys():
                    with st.spinner("Generating profile report..."):
                        generate_profile(df)
                        with open(PROFILE_JSON_PATH, "r") as f:
                            st.session_state.profile_data['dataset'] = json.load(f)
                    if st.session_state.profile_data != {}:
                        st.success("Data profiling complete for the uploaded dataset!")
                    else:
                        st.error("Failed to generate or clean the profile report.")
 
        except Exception as e:
            st.error(f"An error occurred while processing the uploaded file: {e}")
            st.exception(e)
 
    elif uploaded_file is None and st.session_state.loaded_dataframes:
        loaded_tables = list(st.session_state.loaded_dataframes.keys())
        if loaded_tables:
            # selected_table = st.selectbox("Select a table to view its profile:", loaded_tables)
            # if selected_table:
            for table in loaded_tables:
                df_selected = st.session_state.loaded_dataframes[table]
                if table not in st.session_state.profile_data.keys():
                    with st.spinner(f"Generating profile report for table: `{table}`..."):
                        generate_profile(df_selected)
                        with open(PROFILE_JSON_PATH, "r") as f:
                            st.session_state.profile_data[table] = json.load(f)
                # st.session_state.profile_data[table] = generate_profile(df_selected)

            if st.session_state.profile_data != {}:
                st.success(f"Data profiling complete!")
            else:
                st.error(f"Failed to generate or clean the profile report for table: `{table}`.")
        else:
            st.info("No tables are currently loaded from the database.")

if st.session_state.profile_data != {}:
        # if st.button("Suggest Data Quality Rules"):
    for table in st.session_state.profile_data.keys():
        if table not in st.session_state.rules.keys():
            with st.spinner(f"Suggesting Data Quality Rules for `{table}`..."):
                profile_data = st.session_state.profile_data[table].copy()
                del profile_data['missing']
                del profile_data['alerts']
                profile_data['duplicates'] = f"{len(st.session_state.profile_data[table]['duplicates'])} duplicate rows found in data"
                # profile_data = {
                #     'data_types': st.session_state.loaded_dataframes[table].dtypes.to_dict(),
                #     'missing_values': st.session_state.loaded_dataframes[table].isnull().sum().to_dict(),
                #     'unique_values': {col: st.session_state.loaded_dataframes[table][col].value_counts() for col in st.session_state.loaded_dataframes[table].select_dtypes(include=['category']).columns},
                #     'sample_values': {col: st.session_state.loaded_dataframes[table][col].sample(3).tolist() for col in st.session_state.loaded_dataframes[table].columns},
                #     'statistics': st.session_state.loaded_dataframes[table].describe().to_dict()
                # }
                if st.session_state.glossary != {}:
                    st.session_state.rules[table] = create_rules(profile_data, st.session_state.glossary.get(table_name, "N/A"))
                    st.session_state.normalised_rules[table] = normalise_rules(st.session_state.rules[table], st.session_state.glossary[table])
                    # rules[table] = create_rules(profile_data, st.session_state.glossary.get(table_name, "N/A"))
                    # normalised_rules[table] = normalise_rules(st.session_state.rules[table], st.session_state.glossary[table])
                else:
                    st.session_state.rules[table] = create_rules(profile_data)
                    # rules[table] = create_rules(profile_data)
                # if st.session_state.uploaded_rules is not None and table == 'dataset' and st.session_state.uploaded_rules not in st.session_state.rules['dataset']:
                #     st.session_state.rules[table] += "\n" + "**Uploaded Rules**\n" + st.session_state.uploaded_rules

    st.header("Data Profile", divider='violet')
    selected_table = st.selectbox("Select a table to view its profile:", list(st.session_state.loaded_dataframes.keys()))
    if selected_table is not None:
        # --- Sections to Display ---
        sections_to_display = ['table', 'alerts', 'variables', 'correlations', 'missing']
    
        for section_name in sections_to_display:
            st.markdown("---") # Visual separator
    
            if section_name in st.session_state.profile_data[selected_table]:
                section_data = st.session_state.profile_data[selected_table][section_name]
                # st.header(f"`{section_name}`")
    
                # == Table Section Visualization ==
                if section_name == 'table' and isinstance(section_data, dict):
                    st.subheader("Dataset Overview")
                    col1, col2 = st.columns(2)
                    col1.metric("Observations (Rows)", display_value(section_data.get('n', 'N/A')))
                    col1.metric("Variables (Columns)", display_value(section_data.get('n_var', 'N/A')))
                    col1.metric("Missing Cells", display_value(section_data.get('n_cells_missing', 'N/A')))
                    col1.metric("Missing Cells (%)", display_value(section_data.get('p_cells_missing', 'N/A')))
                    col2.metric("Duplicate Rows", display_value(section_data.get('n_duplicates', 'N/A')))
                    col2.metric("Duplicate Rows (%)", display_value(section_data.get('p_duplicates', 'N/A')))
                    mem_size_bytes = section_data.get('memory_size')
                    if isinstance(mem_size_bytes, (int, float)):
                        mem_size_mb = mem_size_bytes / (1024 * 1024)
                        col2.metric("Memory Size", f"{mem_size_mb:,.2f} MiB")
                    else:
                        col2.metric("Memory Size", display_value(mem_size_bytes))
                    col2.metric("Variables with Missing", display_value(section_data.get('n_vars_with_missing', 'N/A')))
                    # if 'types' in section_data and isinstance(section_data['types'], dict):
                    #     st.subheader("Variable Types")
                    #     st.dataframe(pd.Series(section_data['types'], name="Count"))
    
    
                # == Alerts Section Visualization ==
                elif section_name == 'alerts' and isinstance(section_data, list):
                    st.subheader("Data Quality Alerts")
                    if section_data:
                        for alert in section_data:
                            st.info(f"⚠️ {alert}")
                    else:
                        st.info("No alerts for this data.")
    
    
                # == Variables Section Visualization ==
                elif section_name == 'variables' and isinstance(section_data, dict):
                    st.subheader("Column Metrics")
                    st.write(f"Details for **{len(section_data)}** variables:")
                    variable_list = list(section_data.keys())
                    var = st.selectbox("Select a variable to view details:", variable_list, key=f"var_select_{section_name}") # Added unique key
                    # for var in variable_list:
                    if var:
                        # with st.expander(var):
                        var_details = section_data[var]
                        var_type = var_details.get('type', 'N/A')
                        st.markdown(f"#### Details for: `{var}` (Type: {var_type})")
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Distinct Values", display_value(var_details.get('n_distinct', 'N/A')))
                        col1.metric("Unique Values", display_value(var_details.get('n_unique', 'N/A')))
                        col1.metric("Missing Values", display_value(var_details.get('n_missing', 'N/A')))
                        col2.metric("Missing (%)", display_value(var_details.get('p_missing', 'N/A')))
                        # ... (rest of the variable type specific metrics) ...
                        if var_type == "Numeric":
                            col2.metric("Mean", display_value(var_details.get('mean', 'N/A')))
                            col2.metric("Std Dev", display_value(var_details.get('std', 'N/A')))
                            col3.metric("Min", display_value(var_details.get('min', 'N/A')))
                            col3.metric("Max", display_value(var_details.get('max', 'N/A')))
                            col3.metric("Zeros (%)", display_value(var_details.get('p_zeros', 'N/A')))
                        # elif var_type == "Categorical" or var_type == "Boolean":
                        #     if 'value_counts_without_nan' in var_details:
                        #         st.write("**Value Counts:**")
                        #         st.dataframe(pd.Series(var_details['value_counts_without_nan'], name="Count"))
                        elif var_type == "DateTime":
                            col2.metric("Min Date", display_value(var_details.get('min', 'N/A')))
                            col2.metric("Max Date", display_value(var_details.get('max', 'N/A')))
                        elif var_type == "Text":
                            col2.metric("Min Length", display_value(var_details.get('min_length', 'N/A')))
                            col2.metric("Max Length", display_value(var_details.get('max_length', 'N/A')))
                            col3.metric("Mean Length", display_value(var_details.get('mean_length', 'N/A')))

                    # with st.expander("View Full JSON Details for this Variable"):
                    #     st.json(var_details, expanded=False)
    
    
                # == Correlations Section Visualization ==
                elif section_name == 'correlations' and isinstance(section_data, dict):                 
                    if 'auto' in section_data and isinstance(section_data['auto'], list):
                        st.subheader("Correlations")
                        # st.write("**Auto Correlations:** (Attempting DataFrame display)")
                        try:
                            corr_df = pd.DataFrame(section_data['auto'])
                            st.dataframe(corr_df.style.format("{:.2f}").background_gradient(cmap='coolwarm', axis=None))
                        except Exception as e:
                            st.error(f"Could not display 'auto' correlations as DataFrame: {e}")
                            st.write("Showing raw JSON for 'auto' correlations instead:")
                            st.json(section_data['auto'])
                    # else:
                    #     st.write("Raw JSON data for correlations:")
                    #     st.json(section_data)
    
    
                # == Missing Section Visualization (REVISED) ==
                elif section_name == 'missing' and isinstance(section_data, dict):
                    st.subheader("Missing Value Info")
    
                    # --- 1. Nullity Bar Chart (Recreated) ---
                    st.markdown("**Nullity by Column (Missing Value Counts)**")
                    # missing_caption = section_data.get('bar', {}).get('caption', "A simple visualization of nullity by column.")
                    # st.caption(missing_caption)
    
                    # Extract missing counts from the 'variables' section
                    variables_data = st.session_state.profile_data[selected_table].get('variables', {})
                    all_missing_counts = {
                        k: v.get('n_missing', 0)
                        for k, v in variables_data.items()
                        if isinstance(v, dict) and v.get('n_missing', 0) > 0 # Only include if missing > 0
                    }
    
                    if all_missing_counts:
                        # Convert to Series for easy plotting
                        missing_series = pd.Series(all_missing_counts)
                        missing_series = missing_series.sort_values(ascending=False) # Sort descending
    
                        # Create Bar Chart using Matplotlib/Seaborn
                        fig, ax = plt.subplots(figsize=(10, max(4, len(missing_series)*0.3))) # Adjust height based on number of bars
                        sns.barplot(x=missing_series.values, y=missing_series.index, ax=ax, palette="viridis")
                        ax.set_title("Number of Missing Values per Column")
                        ax.set_xlabel("Count of Missing Values")
                        ax.set_ylabel("Column Name")
                        plt.tight_layout() # Adjust layout
                        st.pyplot(fig) # Display the plot in Streamlit
                    else:
                        st.info("No missing values found in any columns.")
    
                    if 'heatmap' in section_data and isinstance(section_data['heatmap'], dict):
                        # with st.expander("View Raw Data for 'Missing Heatmap' (XML - Not Rendered)"):
                        if 'matrix' in section_data['heatmap']:
                            st.subheader("Nullity Correlation")
                            heatmap_caption = section_data.get('heatmap', {}).get('caption', "The correlation heatmap measures nullity correlation...")
                            st.caption(heatmap_caption)
                            # st.code(section_data['heatmap']['matrix'], language='xml')
                            st.image(section_data['heatmap']['matrix'])
                        # else:
                        #     st.json(section_data['heatmap']) # Fallback
                    st.write("-----")
                    # if 'bar' in section_data and isinstance(section_data['bar'], dict):
                    #     # with st.expander("View Raw Data for 'Missing Bar' (XML - Not Rendered)"):
                    #     if 'matrix' in section_data['bar']:
                    #         st.image(section_data['bar']['matrix'])
                    #     else:
                    #         st.json(section_data['bar']) # Fallback
    
    
                # == Default Fallback Display ==
                # else:
                #     st.write("Raw JSON data for this section:")
                #     st.json(section_data)
    
            # else:
            #     st.warning(f"Section **'{section_name}'** not found in the uploaded JSON report.")
    
        # --- Display Duplicates Info ---
        st.markdown("---")
        # st.header("`duplicates`")
        if 'duplicates' in st.session_state.profile_data[selected_table]:
            st.subheader("Duplicates")
            duplicates_data = st.session_state.profile_data[selected_table]['duplicates']
            if isinstance(duplicates_data, list) and duplicates_data:
                # st.subheader("Duplicate Rows Found")
                st.warning(f"Found {len(duplicates_data)} duplicate row entries/indices.")
                with st.expander("View Duplicate Row Details"):
                    st.dataframe(pd.DataFrame(duplicates_data))
            elif 'n_duplicates' in st.session_state.profile_data[selected_table].get('table', {}) and st.session_state.profile_data[selected_table]['table']['n_duplicates'] == 0:
                st.info("No duplicate rows indicated.")
            else:
                st.info("No duplicate row details found in the 'duplicates' section.")
        else:
            st.warning("Section **'duplicates'** not found in the uploaded JSON report.")

        
        # try:
        #     profile_html = st.session_state.profile_data[selected_table].to_html() # Use the passed profile object

        #     st.download_button(
        #         label=f"Download Profile (HTML)",
        #         data=profile_html,
        #         file_name=f"{selected_table} profile.html", # Use the filename parameter here
        #         mime="text/html",
        #         key=f"download {selected_table} profile.html" # Add unique key based on filename
        #     )

        # except Exception as e:
        #     st.error(f"Failed to generate or provide download for HTML report: {e}")   

    # st.header("Data Quality Rules", divider="violet")
    # rules = {}
    # normalised_rules = {}

    # if rules != {}:
    #     for table in rules.keys():
    #         st.session_state.rules[table] = rules[table]
    #         if table in normalised_rules.keys():
    #             st.session_state.normalised_rules[table] = normalised_rules[table]

    if st.session_state.rules != {}:
        if selected_table in st.session_state.rules.keys():
            st.markdown("---")
            st.subheader(f"Data Quality Rules for {table}")
            if selected_table in st.session_state.normalised_rules.keys():
                st.write(st.session_state.normalised_rules[table])
            else:
                st.write(st.session_state.rules[selected_table])

    if st.session_state.rules != {}:
        st.markdown("------")
        if st.button("Cleanse Data"):
            if uploaded_file:
                with st.spinner("Processing..."):
                    # info = {"column datatypes": st.session_state.loaded_dataframes['dataset'].dtypes,
                    #         "sample": st.session_state.loaded_dataframes['dataset'].sample(5)}
                    info = {
                        'data_types': st.session_state.loaded_dataframes[table].dtypes.to_dict(),
                        'missing_values': st.session_state.loaded_dataframes[table].isnull().sum().to_dict(),
                        'unique_categorical_values': {col: st.session_state.loaded_dataframes[table][col].value_counts() for col in st.session_state.loaded_dataframes[table].select_dtypes(include=['category']).columns},
                        'sample_records': st.session_state.loaded_dataframes[table].sample(3)
                    }
                    code = create_pandas(st.session_state.rules, info)
                    
                    is_e = True
                    while is_e:
                        try:
                            df = st.session_state.loaded_dataframes['dataset']
                            local_vars = locals()
                            exec(code, local_vars, None)
                            quarantined_df = local_vars['quarantined_df']
                            validated_df = local_vars['validated_df']
                            st.session_state.cleansed_data['quarantined_dataset'] = quarantined_df
                            st.session_state.cleansed_data['validated_dataset'] = validated_df
                            st.session_state.code = code
                            is_e = False
                        except Exception as e:
                            code = fix_pandas(code, e)
                            continue
                        # st.error(e)
            # else:
            #     code = create_sql(st.session_state.rules, st.session_state.table_ddl)
            #     st.session_state.code = code
            else:
                # --- THIS IS THE MODIFIED BLOCK for database connections ---
                if not st.session_state.rules or not st.session_state.table_ddl:
                    st.warning("No rules generated or table DDL missing. Cannot generate or execute SQL.")
                elif not st.session_state.engine:
                    st.warning("Database connection not established. Cannot execute SQL.")
                else:
                    # Generate SQL
                    with st.spinner("Generating SQL script..."):
                        code = create_sql(st.session_state.rules, st.session_state.table_ddl, st.session_state.engine.dialect)
                        st.session_state.code = code

                    # Execute SQL and Fetch Results if code generation was successful
                    if st.session_state.code:
                        # st.write("Attempting to execute generated SQL on the database...")
                        try:
                            with st.spinner("Executing SQL and fetching results..."):
                                # Execute the generated SQL script
                                with st.session_state.engine.connect() as connection:
                                    # Start transaction
                                    with connection.begin():
                                        # Optional: Add DROP TABLE logic here if you want to allow re-runs
                
                                        # try:
                                        #     connection.execute(text("DROP TABLE IF EXISTS validated_table;"))
                                        #     connection.execute(text("DROP TABLE IF EXISTS quarantined_table;"))
                                        #     # No explicit commit needed here due to transaction context
                                        # except Exception as drop_e:
                                        #     # Log or warn, but proceed maybe?
                                        #     st.warning(f"Could not drop existing tables (they might not exist): {drop_e}")

                                        connection.execute(text(code))
                                    # Transaction implicitly committed here if no errors, or rolled back on error

                                # st.success("SQL script executed successfully.")

                                # Fetch results (first 100 rows)
                                # NOTE: LIMIT syntax might need adjustment for specific DBs (TOP for SQL Server, FETCH FIRST for Oracle)
                                for table_name in st.session_state.loaded_dataframes.keys():
                                    schema, table = table_name.split(".")
                                    validated_query = text(f"SELECT TOP 100 * FROM {schema}.validated_{table}")
                                    quarantined_query = text(f"SELECT TOP 100 * FROM {schema}.quarantined_{table}")

                                    df_validated_sql = pd.read_sql_query(validated_query, st.session_state.engine)
                                    df_quarantined_sql = pd.read_sql_query(quarantined_query, st.session_state.engine)

                                    # Store results in session state using the existing convention
                                    # Use generic names since SQL applies potentially to multiple source tables conceptually
                                    st.session_state.cleansed_data[f'{schema}.validated_{table}'] = df_validated_sql
                                    st.session_state.cleansed_data[f'{schema}.quarantined_{table}'] = df_quarantined_sql
                                    # st.success("Fetched first 100 rows from validated and quarantined tables.")

                        except SQLAlchemyError as db_err:
                            st.error(f"Database error during SQL execution or fetching: {db_err}")
                            # st.code(code, language='sql') # Show the failed SQL
                        except Exception as e:
                            st.error(f"An unexpected error occurred during SQL processing: {e}")
                            # if code:
                            #     st.code(code, language='sql') # Show the potentially problematic SQL
                    else:
                        st.error("SQL code generation failed.")
        
    if st.session_state.cleansed_data != {}:
        st.subheader("Validated Data")
        for key in [key for key in st.session_state.cleansed_data.keys() if "validated_" in key]:    
            st.write(f"**Validated Data for** `{key.replace("validated_", "")}` ({st.session_state.cleansed_data[key].shape[0]})")       
            st.dataframe(st.session_state.cleansed_data[key])
        
        st.subheader("Quarantined Data")
        for key in [key for key in st.session_state.cleansed_data.keys() if "quarantined_" in key]:       
            st.write(f"**Quarantined Data for** `{key.replace("quarantined_", "")}` ({st.session_state.cleansed_data[key].shape[0]})")        
            st.dataframe(st.session_state.cleansed_data[key])
        
        if 'validated_dataset' in st.session_state.cleansed_data and 'quarantined_dataset' in st.session_state.cleansed_data:
            if st.button("Save"):
                with pd.ExcelWriter(f"./output/data quality/{uploaded_file.name.replace(".xlsx", "")} validated.xlsx") as writer:
                    st.session_state.cleansed_data['validated_dataset'].to_excel(writer, sheet_name = "Clean Data")
                with pd.ExcelWriter(f"./output/data quality/{uploaded_file.name.replace(".xlsx", "")} quarantined.xlsx") as writer:
                    st.session_state.cleansed_data['quarantined_dataset'].to_excel(writer, sheet_name = "Exception Data")
                st.success("Cleansed Data saved to **output/data quality**")

    if st.session_state.code is not None:
        st.subheader("Code used for Cleansing Data")
        st.code(st.session_state.code)


else:
    st.info("Awaiting file upload or database connection with selected tables...")