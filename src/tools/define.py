import os

from langchain_core.tools import tool

from src.config.settings import settings
from src.use_cases.define import (
    create_business_glossary_excel as actual_define_create_excel,  # TODO: What do you mean by actual_define_create_excel
)
from src.use_cases.define import (
    parallel_description as actual_define_parallel_description,
)
from src.use_cases.define import (
    read_schema_in_chunks_and_batch as actual_define_read_schema,
)


@tool
def define_data_model_from_excel(excel_path: str, schema_description: str) -> str:
    """
    Generates a business glossary from an input Excel file and a provided schema description.

    Args:
        excel_path (str): Path to the input Excel file containing the data model or schema.
        schema_description (str): A textual description of the schema to guide glossary generation.
    """

    print(
        f"--- TOOL: define_data_model_from_excel called with excel_path: {excel_path} ---"
    )

    if not os.path.exists(excel_path):
        return f"Error: Input Excel file not found at '{excel_path}'."
    if not schema_description:
        return "Error: Schema description is required."

    try:
        # TODO: Why do you have separate DEFINE_OUTPUT_DIR and DEFINE_AI_OUTPUT_DIR
        os.makedirs(settings.DEFINE_OUTPUT_DIR, exist_ok=True)

        _, schema_info_batches = actual_define_read_schema(excel_path)
        all_glossary_parts = []

        for batch in schema_info_batches:
            glossary_part = actual_define_parallel_description(
                schema_description, batch, data_dictionary=[]
            )
            all_glossary_parts.append(glossary_part)

        final_glossary_markdown = "  \n".join(all_glossary_parts)

        # TODO: Why this is hardcoded
        # schema_name_from_path = os.path.basename(excel_path)
        schema_name_from_path = "Schema.xlsx"
        # Note: Your actual_define_create_excel also saves the file. We'll rely on its path logic.
        actual_define_create_excel(schema_name_from_path, final_glossary_markdown)

        output_excel_filename = (
            schema_name_from_path.replace(".xlsx", "") + "_With_Business_Glossary.xlsx"
        )

        output_excel_path = os.path.join(
            settings.DEFINE_OUTPUT_DIR, output_excel_filename
        )

        return f"Business Glossary generation complete. Excel output saved to: {output_excel_path}"
    except Exception as e:
        return f"Error in define_data_model_from_excel: {e}"
