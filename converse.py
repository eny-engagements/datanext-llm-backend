from __future__ import annotations
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Union
import urllib
from timeit import default_timer as timer
import pandas as pd
import openpyxl

import sqlalchemy
from sqlalchemy import (
    MetaData,
    Table,
    create_engine,
    inspect,
    select,
    text,
)
from sqlalchemy.engine import Engine, Result
from sqlalchemy.exc import ProgrammingError, SQLAlchemyError
from sqlalchemy.schema import CreateTable
from sqlalchemy.sql.expression import Executable
from sqlalchemy.types import NullType
from sqlalchemy.engine.url import URL
from sqlalchemy_utils import database_exists
from urllib.parse import quote_plus

from langchain_core._api import deprecated
from langchain_community.utilities import SQLDatabase as _SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain.agents.agent_types import AgentType
from langchain_openai import AzureChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate

import streamlit as st
from streamlit_chat import message


def get_connection_string(config):
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
        return f"Unsupported RDBMS: {config['rdbms']}"
    
    try:
        if config['rdbms'] == "Microsoft SQL Server":
            if config['authentication'] == "Windows Authentication":
                conn_str = f"{drivername}://{config['host']}/{config['database']}?driver=ODBC+Driver+17+for+SQL+Server&Trusted_Connection=yes"
            else:
                conn_str = f"{drivername}://{config['username']}:{config['password']}@{config['host']}/{config['database']}?driver=ODBC+Driver+17+for+SQL+Server"
        elif config['rdbms'] == "OracleDB":
            conn_str = f"{drivername}://{config['username']}:{config['password']}@{config['host']}:{config['port']}/?service_name={config['service']}&encoding=UTF-8&nencoding=UTF-8"
        elif config['rdbms'] == "SQLite":
            config['database'] = config['database'].strip('"').replace('\\','/')
            if database_exists(f"{drivername}:///{config['database']}"):
                conn_str = f"{drivername}:///{config['database']}"
        else:
            conn_str = URL.create(
                drivername=drivername,
                username=config['username'],
                password=config['password'],
                host=config['host'],
                port=config['port'],
                database=config['database']
            )
        return conn_str
    except Exception as e:
        return f"Connection failed: {e}"  

def get_schemas(conn_str):
    try:
        if conn_str.startswith('sqlite:'):
            return ['main']
        
        engine = create_engine(conn_str)
        inspector = inspect(engine)
        schemas = inspector.get_schema_names()

        system_schemas = ['INFORMATION_SCHEMA', 'sys',
                          'information_schema', 'mysql', 'performance_schema',
                          'SYS', 'SYSTEM', 'OUTLN', 'CTXSYS', 'MDSYS']
        
        schemas = [schema for schema in schemas if 
                   schema not in system_schemas 
                   and not schema.startswith('db_') 
                   and not schema.startswith('pg_')]
        
        return schemas
    except Exception as e:
        return f"Error fetching schemas: {e}"  
    
def get_business_glossary(file_path):
    dd_df = pd.read_excel(file_path, engine='openpyxl')
    dd_df['Schema Table'] = dd_df['Schema'] + '.' + dd_df['Table'] 
    table_description = dd_df.groupby('Schema Table')['Table Description'].first().to_dict()
    column_description = dd_df.groupby('Schema Table').apply(
        lambda group: '\n'.join(f"{row['Column']} - {row['Definition']}" for _, row in group.iterrows())
    ).to_dict()
    return table_description, column_description

def _format_index(index: sqlalchemy.engine.interfaces.ReflectedIndex) -> str:
    return (
        f'Name: {index["name"]}, Unique: {index["unique"]},'
        f' Columns: {str(index["column_names"])}'
    )

def truncate_word(content: Any, *, length: int, suffix: str = "...") -> str:
    """
    Truncate a string to a certain number of words, based on the max string
    length.
    """

    if not isinstance(content, str) or length <= 0:
        return content

    if len(content) <= length:
        return content

    return content[: length - len(suffix)].rsplit(" ", 1)[0] + suffix

class SQLDatabase(_SQLDatabase):
    """SQLAlchemy wrapper around a database."""

    def __init__(
        self,
        engine: Engine,
        schemas: List[str],
        metadata: Optional[MetaData] = None,
        ignore_tables: Optional[List[str]] = None,
        include_tables: Optional[List[str]] = None,
        sample_rows_in_table_info: int = 2,
        indexes_in_table_info: bool = False,
        custom_table_info: Optional[dict] = None,
        table_description: Optional[dict] = None,
        column_description: Optional[dict] = None,
        view_support: bool = False,
        max_string_length: int = 300,
        lazy_table_reflection: bool = False,
    ):
        """Create engine from database URI."""
        self._engine = engine
        self._schema = None
        self._schemas = schemas
        if include_tables and ignore_tables:
            raise ValueError("Cannot specify both include_tables and ignore_tables")

        self._inspector = inspect(self._engine)

        self._all_tables_per_schema = {}
        for schema in self._schemas:
            self._all_tables_per_schema[schema] = set(
                self._inspector.get_table_names(schema=schema)
                + (self._inspector.get_view_names(schema=schema) if view_support else [])
            )
        if table_description:
            self._all_tables = set(f"{k}.{name} - {table_description.get(f"{k}.{name}", "")}" for k, names in self._all_tables_per_schema.items() for name in names)
        else:
            self._all_tables = set(f"{k}.{name}" for k, names in self._all_tables_per_schema.items() for name in names)
        

        self._include_tables = set(include_tables) if include_tables else set()
        if self._include_tables:
            missing_tables = self._include_tables - self._all_tables
            if missing_tables:
                raise ValueError(
                    f"include_tables {missing_tables} not found in database"
                )
        self._ignore_tables = set(ignore_tables) if ignore_tables else set()
        if self._ignore_tables:
            missing_tables = self._ignore_tables - self._all_tables
            if missing_tables:
                raise ValueError(
                    f"ignore_tables {missing_tables} not found in database"
                )
        usable_tables = self.get_usable_table_names()
        self._usable_tables = set(usable_tables) if usable_tables else self._all_tables

        if not isinstance(sample_rows_in_table_info, int):
            raise TypeError("sample_rows_in_table_info must be an integer")

        self._sample_rows_in_table_info = sample_rows_in_table_info
        self._indexes_in_table_info = indexes_in_table_info

        self._custom_table_info = custom_table_info
        if self._custom_table_info:
            if not isinstance(self._custom_table_info, dict):
                raise TypeError(
                    "table_info must be a dictionary with table names as keys and the "
                    "desired table info as values"
                )
            # only keep the tables that are also present in the database
            intersection = set(self._custom_table_info).intersection(self._all_tables)
            self._custom_table_info = dict(
                (table, self._custom_table_info[table])
                for table in self._custom_table_info
                if table in intersection
            )

        # Initialising column descriptions dictionary
        self._column_description = column_description or {}

        self._max_string_length = max_string_length
        self._view_support = view_support

        self._metadata = metadata or MetaData()
        if not lazy_table_reflection:
            for schema in self._schemas:
                self._metadata.reflect(
                    views=view_support,
                    bind=self._engine,
                    schema=schema,
                )
        # Add id to tables metadata
        for t in self._metadata.sorted_tables:
            t.id = f"{t.schema}.{t.name}"

    @classmethod
    def from_uri(
        cls, database_uri: str, engine_args: Optional[dict] = None, schemas: Optional[List[str]] = None, **kwargs: Any
    ) -> SQLDatabase:
        """Construct a SQLAlchemy engine from URI."""
        _engine_args = engine_args or {}
        return cls(create_engine(database_uri, **_engine_args), schemas=schemas, **kwargs)

    @property
    def dialect(self) -> str:
        """Return string representation of dialect to use."""
        return self._engine.dialect.name

    def get_usable_table_names(self) -> Iterable[str]:
        """Get names of tables available."""
        if self._include_tables:
            return sorted(self._include_tables)
        return sorted(self._all_tables - self._ignore_tables)

    @deprecated("0.0.1", alternative="get_usable_table_names", removal="0.3.0")
    def get_table_names(self) -> Iterable[str]:
        """Get names of tables available."""
        return self.get_usable_table_names()

    @property
    def table_info(self) -> str:
        """Information about all tables in the database."""
        return self.get_table_info()

    def get_table_info(self, table_names: Optional[List[str]] = None) -> str:
        """Get information about specified tables.

        Follows best practices as specified in: Rajkumar et al, 2022
        (https://arxiv.org/abs/2204.00498)

        If `sample_rows_in_table_info`, the specified number of sample rows will be
        appended to each table description. This can increase performance as
        demonstrated in the paper.
        """
        all_table_names = self.get_usable_table_names()
        all_table_names = [table_name.split(" - ")[0].strip("'") if " - " in table_name else table_name.strip("'") for table_name in all_table_names]

        if table_names is not None:
            table_names = [table_name.split(" - ")[0].strip("'") if " - " in table_name else table_name.strip("'") for table_name in table_names]
            missing_tables = set(table_names).difference(all_table_names)
            if missing_tables:
                raise ValueError(f"table_names {missing_tables} not found in database")
            all_table_names = table_names

        metadata_table_names = [tbl.name for tbl in self._metadata.sorted_tables]
        to_reflect = set(all_table_names) - set(metadata_table_names)
        if to_reflect:
            for schema in self._schemas:
                self._metadata.reflect(
                    views=self._view_support,
                    bind=self._engine,
                    schema=schema,
                )

        meta_tables = [
            tbl
            for tbl in self._metadata.sorted_tables
            if tbl.id in set(all_table_names)
            and not (self.dialect == "sqlite" and tbl.name.startswith("sqlite_"))
        ]

        tables = []
        for table in meta_tables:
            if self._custom_table_info and table.name in self._custom_table_info:
                tables.append(self._custom_table_info[table.name])
                continue

            # Ignore JSON datatyped columns
            for k, v in table.columns.items():
                if type(v.type) is NullType:
                    table._columns.remove(v)

            # add create table command
            create_table = str(CreateTable(table).compile(self._engine))
            table_info = f"{create_table.rstrip()}"
            column_desc = self._column_description.get(table.id)
            if column_desc:
                table_info += "\n\n/* Column Descriptions:\n"
                table_info += column_desc
                table_info += "\n*/"     
            has_extra_info = (
                self._indexes_in_table_info or self._sample_rows_in_table_info
            )
            if has_extra_info:
                table_info += "\n\n/*"
            if self._indexes_in_table_info:
                table_info += f"\n{self._get_table_indexes(table)}\n"
            if self._sample_rows_in_table_info:
                table_info += f"\n{self._get_sample_rows(table)}\n"
            if has_extra_info:
                table_info += "*/"

            tables.append(table_info)
            
        tables.sort()
        final_str = "\n\n".join(tables)
        return final_str

    def _get_table_indexes(self, table: Table) -> str:
        indexes = self._inspector.get_indexes(table.name)
        indexes_formatted = "\n".join(map(_format_index, indexes))
        return f"Table Indexes:\n{indexes_formatted}"

    def _get_sample_rows(self, table: Table) -> str:
        # build the select command
        command = select(table).limit(self._sample_rows_in_table_info)

        # save the columns in string format
        columns_str = "\t".join([col.name for col in table.columns])

        try:
            # get the sample rows
            with self._engine.connect() as connection:
                sample_rows_result = connection.execute(command)  # type: ignore
                # shorten values in the sample rows
                sample_rows = list(
                    map(lambda ls: [str(i)[:100] for i in ls], sample_rows_result)
                )

            # save the sample rows in string format
            sample_rows_str = "\n".join(["\t".join(row) for row in sample_rows])

        # in some dialects when there are no rows in the table a
        # 'ProgrammingError' is returned
        except ProgrammingError:
            sample_rows_str = ""

        return (
            f"{self._sample_rows_in_table_info} rows from {table.name} table:\n"
            f"{columns_str}\n"
            f"{sample_rows_str}"
        )

    def _execute(
        self,
        command: Union[str, Executable],
        fetch: Literal["all", "one", "cursor"] = "all",
        *,
        parameters: Optional[Dict[str, Any]] = None,
        execution_options: Optional[Dict[str, Any]] = None,
    ) -> Union[Sequence[Dict[str, Any]], Result]:
        """
        Executes SQL command through underlying engine.

        If the statement returns no rows, an empty list is returned.
        """
        parameters = parameters or {}
        execution_options = execution_options or {}
        with self._engine.begin() as connection:  # type: Connection  # type: ignore[name-defined]
            if self._schema is not None:
                if self.dialect == "snowflake":
                    connection.exec_driver_sql(
                        "ALTER SESSION SET search_path = %s",
                        (self._schema,),
                        execution_options=execution_options,
                    )
                elif self.dialect == "bigquery":
                    connection.exec_driver_sql(
                        "SET @@dataset_id=?",
                        (self._schema,),
                        execution_options=execution_options,
                    )
                elif self.dialect == "mssql":
                    pass
                elif self.dialect == "trino":
                    connection.exec_driver_sql(
                        "USE ?",
                        (self._schema,),
                        execution_options=execution_options,
                    )
                elif self.dialect == "duckdb":
                    # Unclear which parameterized argument syntax duckdb supports.
                    # The docs for the duckdb client say they support multiple,
                    # but `duckdb_engine` seemed to struggle with all of them:
                    # https://github.com/Mause/duckdb_engine/issues/796
                    connection.exec_driver_sql(
                        f"SET search_path TO {self._schema}",
                        execution_options=execution_options,
                    )
                elif self.dialect == "oracle":
                    connection.exec_driver_sql(
                        f"ALTER SESSION SET CURRENT_SCHEMA = {self._schema}",
                        execution_options=execution_options,
                    )
                elif self.dialect == "sqlany":
                    # If anybody using Sybase SQL anywhere database then it should not
                    # go to else condition. It should be same as mssql.
                    pass
                elif self.dialect == "postgresql":  # postgresql
                    connection.exec_driver_sql(
                        "SET search_path TO %s",
                        (self._schema,),
                        execution_options=execution_options,
                    )

            if isinstance(command, str):
                command = text(command)
            elif isinstance(command, Executable):
                pass
            else:
                raise TypeError(f"Query expression has unknown type: {type(command)}")
            cursor = connection.execute(
                command,
                parameters,
                execution_options=execution_options,
            )

            if cursor.returns_rows:
                if fetch == "all":
                    result = [x._asdict() for x in cursor.fetchall()]
                elif fetch == "one":
                    first_result = cursor.fetchone()
                    result = [] if first_result is None else [first_result._asdict()]
                elif fetch == "cursor":
                    return cursor
                else:
                    raise ValueError(
                        "Fetch parameter must be either 'one', 'all', or 'cursor'"
                    )
                return result
        return []

    def run(
        self,
        command: Union[str, Executable],
        fetch: Literal["all", "one", "cursor"] = "all",
        include_columns: bool = False,
        *,
        parameters: Optional[Dict[str, Any]] = None,
        execution_options: Optional[Dict[str, Any]] = None,
    ) -> Union[str, Sequence[Dict[str, Any]], Result[Any]]:
        """Execute a SQL command and return a string representing the results.

        If the statement returns rows, a string of the results is returned.
        If the statement returns no rows, an empty string is returned.
        """
        result = self._execute(
            command, fetch, parameters=parameters, execution_options=execution_options
        )

        if fetch == "cursor":
            return result

        res = [
            {
                column: truncate_word(value, length=self._max_string_length)
                for column, value in r.items()
            }
            for r in result
        ]

        if not include_columns:
            res = [tuple(row.values()) for row in res]  # type: ignore[misc]

        if not res:
            return ""
        else:
            return str(res)

    def get_table_info_no_throw(self, table_names: Optional[List[str]] = None) -> str:
        """Get information about specified tables.

        Follows best practices as specified in: Rajkumar et al, 2022
        (https://arxiv.org/abs/2204.00498)

        If `sample_rows_in_table_info`, the specified number of sample rows will be
        appended to each table description. This can increase performance as
        demonstrated in the paper.
        """
        try:
            return self.get_table_info(table_names)
        except ValueError as e:
            """Format the error message"""
            return f"Error: {e}"

    def run_no_throw(
        self,
        command: str,
        fetch: Literal["all", "one"] = "all",
        include_columns: bool = False,
        *,
        parameters: Optional[Dict[str, Any]] = None,
        execution_options: Optional[Dict[str, Any]] = None,
    ) -> Union[str, Sequence[Dict[str, Any]], Result[Any]]:
        """Execute a SQL command and return a string representing the results.

        If the statement returns rows, a string of the results is returned.
        If the statement returns no rows, an empty string is returned.

        If the statement throws an error, the error message is returned.
        """
        try:
            return self.run(
                command,
                fetch,
                parameters=parameters,
                execution_options=execution_options,
                include_columns=include_columns,
            )
        except SQLAlchemyError as e:
            """Format the error message"""
            return f"Error: {e}"

    def get_context(self) -> Dict[str, Any]:
        """Return db context that you may want in agent prompt."""
        table_names = list(self.get_usable_table_names())
        table_info = self.get_table_info_no_throw()
        return {"table_info": table_info, "table_names": ", ".join(table_names)}
    
def query_rdbms(user_input, conn_str, schemas, table_description=None, column_description=None):
    db = SQLDatabase.from_uri(conn_str, schemas=schemas, view_support=False, table_description=table_description, column_description=column_description)

    llm = AzureChatOpenAI(
        azure_deployment="gpt-4o",
        openai_api_version="2024-02-15-preview",
        api_key="901cce4a4b4045b39727bf1ce0e36219",
        azure_endpoint="https://aifordata-azureopenai.openai.azure.com/",
        temperature=0,
        max_tokens=4096
    )

    final_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", 
             """
              You are a helpful AI assistant expert in querying SQL Database to find answers to user's question.
              A general guideline is to limit the number of results from a query to 10 unless the user question
              specifies otherwise.
              When invoking sql_db_schema, prefix the table name with the schema name.
              Ensure the query uses the correct functions and keywords as per the {dialect} dialect.
              In cases where the names of tables and columns are codified and ambiguous, a description for them
              will be provided. Follow those definitions to join the correct tables in your query. Augment the 
              generation of the query by understanding the column descriptions provided after the DDL of the tables
              and be very careful about using the correct columns, and joining tables with the accurate primary key
              and foreign key constraints.
              Only provide the precise answer of the question and nothing else.
              If a query does not return any results despite being syntactically correct, try a different query which
              may or may not involve different tables than the previous queries.
              I will set the max iterations. If the last iteration of querying the database returns no results, STRICTLY
              MENTION "I dont know" and nothing else. STRICTLY DO NOT provide any results not obtained from the query.
              STRICTLY DO NOT display name of any database, schema, table, view or column in the answer.
              STRICTLY DO NOT add any Note or Summary at the bottom.
             """
             ),
            ("user", "{question}\n ai: "),
        ]
    )

    agent_executor = create_sql_agent(llm, db=db, agent_type="openai-tools", verbose=True, max_iterations=5)
    response = agent_executor.invoke(final_prompt.format(question=user_input, dialect=db.dialect))

    return response['output']


# Streamlit interface
st.set_page_config(layout="wide",
                   page_title="Converse with Data",
                   page_icon="💬" 
)

st.subheader("Converse with Data Sources using Generative AI and Obtain Insights Conveniently💬")
st.markdown(
"""
The accelerator makes analytics seamless. Type your question and let Generative AI handle the aggregation and querying for you.

----
""")

if "databases" not in st.session_state:
    st.session_state.databases = []
if "data_source" not in st.session_state:
    st.session_state.data_source = {}
if "user_msgs" not in st.session_state:
    st.session_state.user_msgs = []
if "system_msgs" not in st.session_state:
    st.session_state.system_msgs = []


with st.sidebar:
    with st.expander("Connect to Database"):
        db_type = st.selectbox("Database Type", ["PostgreSQL", "MySQL", "Microsoft SQL Server", "OracleDB", "SQLite"])
        
        connection_details = {}
        if db_type == "SQLite":
            connection_details['database'] = st.text_input("Database File Path")
        else:
            connection_details['host'] = st.text_input("Host", type="password")
            connection_details['username'] = st.text_input("Username", type="password")
            connection_details['password'] = st.text_input("Password", type="password")
            connection_details['business_glossary'] = st.text_input("Path to Business Glossary")
            connection_details['database'] = st.text_input("Database Name", type="password")
            if db_type in ["PostgreSQL", "MySQL"]:
                connection_details['port'] = st.text_input("Port")
            if db_type == "OracleDB":
                connection_details['service'] = st.text_input("Service Name")
            if db_type == "Microsoft SQL Server":
                connection_details['authentication'] = st.selectbox("Authentication Type", ["Windows Authentication", "Others"])
        
        if st.button("Add Data Source"):
            config = {
                "rdbms": db_type,
                **connection_details
            }
            database_name = f"{db_type} > {connection_details['database']}"
            st.session_state.databases.append(database_name)
            conn_str = get_connection_string(config)
            schemas = get_schemas(conn_str)
            table_description, column_description = get_business_glossary(connection_details['business_glossary'].strip('"')) if connection_details['business_glossary'].strip() != "" else (None, None)
            st.session_state.data_source[database_name] = {
                'connection_string': conn_str,
                'schemas': schemas,
                'table_description': table_description,
                'column_description': column_description
            }

            st.success("Database added successfully")
    
    # with st.expander("Add Data Files"):
    #     uploaded_files = st.file_uploader("Add Data Files", accept_multiple_files=True, type=['json', 'csv', 'xlsx'])
    #     if uploaded_files:
    #         for uploaded_file in uploaded_files:
    #             if uploaded_file.name.endswith(".csv"):



with st.form("user_input_form", clear_on_submit=True):
    user_input = st.text_input("Enter your question", key="input")
    db = st.selectbox("Database to Query", st.session_state.databases)
    submitted = st.form_submit_button("Submit")

    if submitted:
        with st.spinner("Processing your question..."):
            st.session_state.user_msgs.append(user_input)
            conn_str = st.session_state.data_source[db]['connection_string']
            schemas = st.session_state.data_source[db]['schemas']
            table_description = st.session_state.data_source[db]['table_description']
            column_description = st.session_state.data_source[db]['column_description']
            start = timer()

            try:
                answer = query_rdbms(user_input, conn_str, schemas, table_description=table_description, column_description=column_description)
                st.session_state.system_msgs.append(answer)

            except Exception as e:
                st.write(f"Failed to process question. Please try again. {e}")
                print(e)

        st.write(f"Time taken: {timer() - start:.2f}s")
    
if st.session_state["system_msgs"]:
    for i in range(len(st.session_state["system_msgs"]) - 1, -1, -1):
        message(st.session_state["system_msgs"][i], key = str(i) + "_assistant")
        message(st.session_state["user_msgs"][i], is_user=True, key=str(i) + "_user")