import streamlit as st

import os
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

import openpyxl
from openpyxl import load_workbook
from openpyxl.styles import Alignment, PatternFill, Border, Side

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document

from langchain_openai import AzureChatOpenAI
# from langchain_ollama.llms import OllamaLLM 
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


# Initialise LLM
llm = AzureChatOpenAI(
    azure_deployment="gpt-4o",
    openai_api_version="2024-02-15-preview",
    api_key="901cce4a4b4045b39727bf1ce0e36219",
    azure_endpoint="https://aifordata-azureopenai.openai.azure.com/",
    temperature=0.5,
    max_tokens=4096
    )

# llm = OllamaLLM(model="llama3.3:70b-instruct-q8_0")

def read_schema(schema_file, batch_size=20):
    schema_df = pd.read_excel(schema_file, engine='openpyxl')
    schema_df = schema_df.drop_duplicates(subset=['Table Name', 'Column Name'], keep='first')
    schema_df = schema_df[schema_df['Table Name'].notna() & schema_df['Column Name'].notna()]
    schema_df = schema_df[schema_df['Schema Name'].notna() & schema_df['Table Name'].notna()]
    schema_df = schema_df[schema_df['Column Name'].str.strip() != '']
    schema_df = schema_df[schema_df['Table Name'].str.strip() != '']
    schema_df = schema_df[schema_df['Schema Name'].str.strip() != '']

    schema_info_batches = []
    current_batch = []
    has_column_name = 'Column Name' in schema_df.columns
    has_ddl = 'DDL' in schema_df.columns

    for schema in schema_df['Schema Name'].unique():
        schema_tables = schema_df[schema_df['Schema Name'] == schema]['Table Name'].unique()
        for table in schema_tables:
            chunk = f"\nSource System: {schema}\nTable: {table}\n"
            if has_ddl:
                ddl_val = schema_df[(schema_df['Schema Name'] == schema) & (schema_df['Table Name'] == table)]['DDL'].values[0]
                chunk += f"DDL: {ddl_val}\n"
            if has_column_name:
                columns = [
                    " ".join(col.split())
                    for col in schema_df[
                        (schema_df['Schema Name'] == schema) & (schema_df['Table Name'] == table)
                    ]['Column Name'].astype(str).tolist()
                ]
                if columns:
                    chunk += "Columns: " + ", ".join(columns) + "\n"
            current_batch.append(chunk)
            if len(current_batch) == batch_size:
                schema_info_batches.append(current_batch)
                current_batch = []
    if current_batch:
        schema_info_batches.append(current_batch)

    return schema_df, schema_info_batches

def ddl_description(chunk, schema_description):
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
    4. Only describe the tables and columns listed below. Do not invent or add any that are not present.

    Schema Description:
    {schema_description}
    
    Table Info: 
    {schema_data}

    Use this format throughout:
    Table: [Table Name]\n
    Description: [Concise description of the table]\n                            
    - [Column 1 Name]: [Comprehensive and business-centric definition for the column]\n  
    - [Column 2 Name]: [Comprehensive and business-centric definition for the column]\n 
    ...
    - [Column n Name]: [Comprehensive and business-centric definition for the column] 
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
    4. Only describe the tables and columns listed below. Do not invent or add any that are not present.

    Schema Description:
    {schema_description}

    Table Info: 
    {schema_data}

    Glossary generated so far:
    {glossary}

    Continue from the last point and complete the glossary by providing definitions for the remaining columns.

    Use this format throughout:                          
    - [Remaining Column 1 Name]: [Comprehensive and business-centric definition for the column]\n  
    - [Column 2 Name]: [Comprehensive and business-centric definition for the column]\n 
    ...
    - [Column n Name]: [Comprehensive and business-centric definition for the column] 
    ------
                                                    
    Do not include introductory or closing messages. After describing all columns, end your response with ------.
    """)

    runnable = RunnablePassthrough()
    output_parser = StrOutputParser()

    chain = runnable | db_schema_prompt | llm | output_parser
    continue_chain = runnable | db_schema_continue_prompt | llm | output_parser

    glossary = chain.invoke({"schema_data": chunk, "schema_description": schema_description})

    while not glossary.endswith("------"):
        glossary_lines = glossary.split('\n')
        if not chunk.split(", ")[-1] in glossary_lines[-1]:
            glossary = '\n'.join(glossary_lines[:-1])
        else:
            glossary_lines[-1] = glossary_lines[-1].split(": ")[0] + "Could not be generated.\n------"
            glossary = '\n'.join(glossary_lines)
            break
        continued_glossary = continue_chain.invoke({"schema_data": chunk, "glossary": glossary, "schema_description": schema_description})
        glossary += f"\n{continued_glossary}"

    return glossary

def generate_glossary_ddl(schema_info, schema_description):
    glossaries = []
    futures = []
    with ThreadPoolExecutor(max_workers=20) as executor:
        for chunk in schema_info:
            future = executor.submit(ddl_description, chunk, schema_description)
            futures.append(future)

    glossaries.extend([future.result() for future in futures])

    glossary = "  \n".join(glossaries)
    return glossary
    

def create_business_glossary_excel(schema_name, glossary):
    business_glossary_file_path = schema_name.replace(".xlsx", "") + "_with_Business_Glossary.xlsx"
    sheet_name = "Business Glossary"

    lines = glossary.strip().split('\n')
    tables = []
    current_table = {}
    for line in lines:
        if line.startswith('Table:'):
            if current_table:
                tables.append(current_table)
            current_table = {
                'table_name': line.split('Table:')[1].strip(),
                'description': '',
                'columns': []
            }
        elif line.startswith('Description:'):
            current_table['description'] = line.split('Description:')[1].strip()
        elif line.startswith('-'):
            column_detail = line.split(':', 1)
            if len(column_detail) > 1:
                column_name = column_detail[0].strip('-').strip()
                column_description = column_detail[1].strip()
                current_table['columns'].append((column_name, column_description))
            else:
                print(f"Skipping improperly formatted line: {line.strip()}")

    if current_table:
        tables.append(current_table)

    rows = []
    for table in tables:
        table_name = table['table_name']
        table_description = table['description']
        columns = table['columns']
        for column_name, column_description in columns:
            rows.append([table_name, table_description, column_name, column_description])
    
    schema_df = pd.DataFrame(rows, columns=['Table Name', 'Table Description', 'Column Name', 'Column Description'])
    valid_tables = set(zip(st.session_state['schema_df']['Table Name'], st.session_state['schema_df']['Column Name']))
    filtered_rows = [row for row in rows if (row[0], row[2]) in valid_tables]
    schema_df = pd.DataFrame(filtered_rows, columns=['Table Name', 'Table Description', 'Column Name', 'Column Description'])
    schema_df = schema_df.drop_duplicates(subset=['Table Name', 'Column Name'], keep='first')
    if not os.path.exists(f"./output/business glossary"):
        os.makedirs(f"./output/business glossary")
    schema_df.to_excel(f"./output/business glossary/{business_glossary_file_path}", sheet_name=sheet_name, index=False)

    workbook = load_workbook(f"./output/business glossary/{business_glossary_file_path}")
    sheet = workbook[sheet_name]

    # Set column widths
    sheet.column_dimensions['A'].width = 30
    sheet.column_dimensions['B'].width = 36
    sheet.column_dimensions['C'].width = 30
    sheet.column_dimensions['D'].width = 36

    # Freeze top row
    sheet.freeze_panes = sheet['A2']

    # Define border style
    thin = Side(border_style="thin", color="000000")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)

    # Set alignment and borders for all cells, and set row height for header
    for row in sheet.iter_rows(min_row=1, max_row=sheet.max_row, min_col=1, max_col=sheet.max_column):
        for idx, cell in enumerate(row):
            # Word wrap and vertical middle for all
            cell.alignment = Alignment(wrap_text=True, vertical='center')
            # Borders for all
            cell.border = border
            # Header formatting
            if cell.row == 1:
                cell.alignment = Alignment(horizontal='center', vertical='center', wrap_text=True)
        if row[0].row == 1:
            sheet.row_dimensions[1].height = 30
        else:
            # Left justify A, C, D; Center B
            if len(row) >= 1:
                row[0].alignment = Alignment(horizontal='left', vertical='center', wrap_text=True)
            if len(row) >= 2:
                row[1].alignment = Alignment(horizontal='center', vertical='center', wrap_text=True)
            if len(row) >= 3:
                row[2].alignment = Alignment(horizontal='left', vertical='center', wrap_text=True)
            if len(row) >= 4:
                row[3].alignment = Alignment(horizontal='left', vertical='center', wrap_text=True)

    start_row = 2 
    col_a_letter = 'A'
    col_b_letter = 'B'

    row = start_row
    while row <= sheet.max_row:
        cell_a = sheet[f'{col_a_letter}{row}']
        cell_b = sheet[f'{col_b_letter}{row}']
        
        if cell_a.value is not None:
            merge_start = row
            merge_end = row
            
            while merge_end < sheet.max_row and sheet[f'{col_a_letter}{merge_end + 1}'].value == cell_a.value:
                merge_end += 1
            
            if merge_end > merge_start:
                sheet.merge_cells(start_row=merge_start, start_column=cell_b.column, end_row=merge_end, end_column=cell_b.column)
                merged_cell = sheet.cell(row=merge_start, column=cell_b.column)
                merged_cell.alignment = Alignment(wrapText=True, vertical='top')
            
            row = merge_end + 1
        else:
            row += 1

    header_fill = PatternFill(start_color="D0E8FA", end_color="D0E8FA", fill_type="solid")
    for cell in sheet[1]:
        cell.fill = header_fill

    workbook.save(f"./output/business glossary/{business_glossary_file_path}")
    return f"./output/business glossary/{business_glossary_file_path}"


# Streamlit Interface
if 'schema_info' not in st.session_state:
    st.session_state['schema_info'] = ""
if 'schema_info_batches' not in st.session_state:
    st.session_state['schema_info_batches'] = []
if 'schema_df' not in st.session_state:  
    st.session_state['schema_df'] = None
if 'glossary' not in st.session_state:
    st.session_state['glossary'] = ""
if 'glossary_continuation' not in st.session_state:
    st.session_state['glossary_continuation'] = ""


st.set_page_config(
    layout="wide",
    page_title="Automatic Generation Business Glossary",
    page_icon="📖",
)

st.subheader("Automatic Creation of Business Glossary using EY's Expertise and Large Language Models 📖")
st.markdown(
"""
The accelerator supports teams in cataloging source systems which may often contain ambiguous names for entities and attributes. 
It resolves such ambiguities by leveraging EY's extensive experience in the financial and life insurance sector. It accepts the 
path to any excel sheet containing a list of table names and column names and a brief description of the source system where you
may include further specifications regarding updates and naming conventions. The accelerator relies on a RAG pipeline to a knowledge 
base and a Large Language Model.

The outcome is an excel sheet containing table and column descriptions, which can be found in the **output/business glossary** directory.

----
""")

schema_file = st.file_uploader("Upload Schema File (Should contain fields: Schema Name, Table Name and Column Name)", type=["xlsx"])
schema_description = st.text_area("Enter the schema description:")

if schema_file:
    schema_df, schema_info_batches = read_schema(schema_file)
    st.session_state['schema_info_batches'] = schema_info_batches
    st.session_state['schema_df'] = schema_df

if schema_description and st.session_state['schema_info_batches'] != []:
    schema_name = schema_file.name

    st.subheader("Tables and Columns", divider='violet')
    for batch in st.session_state['schema_info_batches']:
        show_schema = ''.join(batch)
        st.text(show_schema)

    if st.button("Describe"):
        for batch_index, batch in enumerate(st.session_state['schema_info_batches']):
            try:
                with st.spinner(f"Generating Business Glossary for batch {batch_index + 1} of {len(st.session_state['schema_info_batches'])}"):
                    glossary = generate_glossary_ddl(batch, schema_description)
                    st.session_state['glossary_continuation'] = f"  \n{glossary}"
                    st.session_state['glossary'] += st.session_state['glossary_continuation']
                st.markdown(st.session_state['glossary_continuation'])
            except Exception as e:
                print(f"Exception at batch {batch_index + 1}: ", e)
                continue
    
    if st.button("Create Business Glossary Excel Sheet") and st.session_state['glossary'] != "":
        try:
            file_path = create_business_glossary_excel(schema_name, st.session_state['glossary'])
            st.success("Business Glossary Excel sheet ready for download! Please proceed")
            st.session_state['glossary'] = ""
            st.session_state['glossary_continuation'] = ""
        except Exception as e:
            st.error(f"Error creating Business Glossary Excel sheet: {e}")
            file_path = None

        with open(file_path, "rb") as f:
            st.download_button(
                label="Download Business Glossary Excel Sheet",
                data=f,
                file_name=os.path.basename(file_path),
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                on_click=lambda: os.remove(file_path)
            )
