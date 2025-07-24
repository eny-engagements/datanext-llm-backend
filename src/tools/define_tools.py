from use_cases.define import read_schema_in_chunks_and_batch as actual_define_read_schema
from use_cases.define import parallel_description as actual_define_parallel_description
from use_cases.define import create_business_glossary_excel as actual_define_create_excel
import pandas as pd
import os
import shutil
import asyncio
from langchain_core.tools import tool
from typing import Optional

DEFINE_AI_OUTPUT_DIR = "C:/EY/ey-genai-datanext-frontend/public"

os.makedirs(DEFINE_AI_OUTPUT_DIR, exist_ok=True)




@tool
def define_data_model_from_excel(excel_path: str, schema_description: str) -> str:
    """
    Creates a business glossary from an input Excel file...
    """
    print(f"--- TOOL: define_data_model_from_excel called with excel_path: {excel_path} ---")
    if not os.path.exists(excel_path): return f"Error: Input Excel file not found at '{excel_path}'."
    if not schema_description: return "Error: Schema description is required."

    try:
        # --- FIX: Ensure output directory exists ---
        os.makedirs(DEFINE_AI_OUTPUT_DIR, exist_ok=True)
        # --- END FIX ---

        _, schema_info_batches = actual_define_read_schema(excel_path)
        all_glossary_parts = []
        for batch in schema_info_batches:
            glossary_part = actual_define_parallel_description(schema_description, batch, data_dictionary=[])
            all_glossary_parts.append(glossary_part)
        final_glossary_markdown = "  \n".join(all_glossary_parts)

        # schema_name_from_path = os.path.basename(excel_path)
        schema_name_from_path = "Schema.xlsx"
        # Note: Your actual_define_create_excel also saves the file. We'll rely on its path logic.
        actual_define_create_excel(schema_name_from_path, final_glossary_markdown)

        output_excel_filename = schema_name_from_path.replace(".xlsx", "") + "_With_Business_Glossary.xlsx"
        output_excel_path = os.path.join(DEFINE_AI_OUTPUT_DIR, output_excel_filename)

        return f"Business Glossary generation complete. Excel output saved to: {output_excel_path}"
    except Exception as e:
        return f"Error in define_data_model_from_excel: {e}"
        # db_name_for_file = database or service or "discovered_db"
        # output_csv_filename = f"discovered_glossary_{db_name_for_file}.csv"
        # output_csv_path = os.path.join(DISCOVER_AI_OUTPUT_DIR, output_csv_filename)
        # with open(output_csv_path, "w", newline="") as f:
        #     f.write(csv_content)
    
