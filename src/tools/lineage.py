import os
from typing import Optional

from langchain_core.tools import tool

from src.config.settings import settings

# TODO: what are these function names??
from src.use_cases.lineage import analyse_sp as actual_lineage_extractor_analyse_sp
from src.use_cases.lineage import (
    create_etl_documentation_excel as actual_lineage_extractor_create_excel,
)
from src.use_cases.lineage import get_glossary as actual_lineage_extractor_get_glossary
from src.use_cases.lineage import (
    visualise_etl as actual_lineage_extractor_visualise_etl,
)

# TODO: Why do you have separate LINEAGE_OUTPUT_DIR and LINEAGE_AI_OUTPUT_DIR
os.makedirs(settings.LINEAGE_AI_OUTPUT_DIR, exist_ok=True)


@tool
def extract_etl_lineage_and_documentation(
    script_path: str,
    dialect: str,
    additional_context: Optional[str] = "N/A",
    business_glossary_excel_path: Optional[str] = None,
) -> str:
    """
    Analyze an ETL script to generate documentation, a data lineage diagram, and an Excel report.

    Args:
        script_path (str): Path to the ETL script file (SQL or text file).
        dialect (str): SQL dialect of the script (e.g., 'T-SQL', 'Spark SQL').
        additional_context (Optional[str], optional): Additional natural language context to assist analysis. Defaults to "N/A".
        business_glossary_excel_path (Optional[str], optional): Path to a business glossary Excel file to enrich the analysis. Defaults to None.
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
        os.makedirs(settings.LINEAGE_AI_OUTPUT_DIR, exist_ok=True)

        # TODO: Why is it hardcoded?
        # report_name = os.path.basename(script_path)
        report_name = "Lineage.txt"
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
            ".txt", "_ETL_Documentation.xlsx"
        ).replace(".sql", "_ETL_Documentation.xlsx")
        output_excel_path = os.path.join(
            settings.LINEAGE_AI_OUTPUT_DIR, output_excel_filename
        )

        return (
            f"ETL Lineage and Documentation generation complete. "
            f"Lineage Diagram saved to: {diagram_path}. "
            f"Excel documentation saved to: {output_excel_path}."
        )
    except Exception as e:
        return f"Error in extract_etl_lineage_and_documentation: {e}"
