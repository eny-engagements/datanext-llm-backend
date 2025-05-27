import streamlit as st
from lida import Manager, TextGenerationConfig, llm
from lida.datamodel import Goal
import os
import pandas as pd
import json
import urllib
from converse import SQLDatabase, get_business_glossary
# from langchain.agents import create_sql_agent
# from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_openai import AzureChatOpenAI
from langchain.agents.agent_types import AgentType  
from streamlit_chat import message
from timeit import default_timer as timer
from langchain.prompts.chat import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.prompts.prompt import PromptTemplate
from sqlalchemy import create_engine

# Initialize selected_dataset to None
selected_dataset = None
relevant_table_count = 0
esg_compass_relevant_datasets = {}

# Functions

def get_relevant_tables(str):
    return str.split(",")

# make data dir if it doesn't exist
os.makedirs("data", exist_ok=True)

st.set_page_config(
    layout="wide",
    page_title="Automatic Generation of Visualization Goals",
    page_icon="📊",
)

st.write("### Automatic Generation of Visualization Goals and its Specifications using Large Language Models 📊")

st.sidebar.write("## Setup")


openai_key = "901cce4a4b4045b39727bf1ce0e36219"
azure_endpoint = "https://aifordata-azureopenai.openai.azure.com/"
openai_api_version = "2024-05-01-preview"

st.markdown(
"""
The system supports users in the automatic creation of visualizations and addresses several subtasks such as understand the semantics of data, enumerate     relevant visualization goals and generate visualization specifications. It enumerates visualization goals given the data and generates, refines,         executes and filters visualization. 

This can work seamlessly either on any data file like CSV, JSON or any RDBMS.

   ----
""")

# Step 2 - Select a dataset and summarization method
if openai_key:
    
    # select model from gpt-4 , gpt-3.5-turbo, gpt-3.5-turbo-16k
    st.sidebar.write("## Text Generation Model")
    #models = ["gpt-4-32k"]
    models = ["gpt-4o"]
    selected_model = st.sidebar.selectbox(
        'Choose a model',
        options=models,
        index=0
    )

    # select temperature on a scale of 0.0 to 1.0
    # st.sidebar.write("## Text Generation Temperature")
    temperature = st.sidebar.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.0)

    # set use_cache in sidebar
    use_cache = st.sidebar.checkbox("Use cache", value=True)

    st.sidebar.write("### Choose a summarization method")
    # summarization_methods = ["default", "llm", "columns"]
    summarization_methods = [
        {"label": "llm",
         "description":
         "Uses the LLM to generate annotate the default summary, adding details such as semantic types for columns and dataset description"},
        {"label": "default",
         "description": "Uses dataset column statistics and column names as the summary"},

        {"label": "columns", "description": "Uses the dataset column names as the summary"}]

    # selected_method = st.sidebar.selectbox("Choose a method", options=summarization_methods)
    selected_method_label = st.sidebar.selectbox(
        'Choose a method',
        options=[method["label"] for method in summarization_methods],
        index=0
    )

    selected_method = summarization_methods[[
        method["label"] for method in summarization_methods].index(selected_method_label)]["label"]

    # add description of selected method in very small font to sidebar
    selected_summary_method_description = summarization_methods[[
        method["label"] for method in summarization_methods].index(selected_method_label)]["description"]

    if selected_method:
        st.sidebar.markdown(
            f"<span> {selected_summary_method_description} </span>",
            unsafe_allow_html=True)

    index = 0
    num_goals = 0
    own_goal = ""
    user_goal = ""
    selected_goal = 0
    selected_library = ""
    num_visualizations = 0
    selected_viz_title = ""
    
    st.sidebar.write("### Goal Selection")
    num_goals = st.sidebar.slider("Number of goals to generate", min_value=1, max_value=10, value=2)
    #own_goal = st.sidebar.checkbox("Add Your Own Goal")
    #if own_goal:
        #user_goal = st.sidebar.text_input("Describe Your Goal")
    st.sidebar.write("## Visualization Library")
    visualization_libraries = ["seaborn", "matplotlib", "plotly"]
    selected_library = st.sidebar.selectbox('Choose a visualization library', options=visualization_libraries, index=0)
    num_visualizations = st.sidebar.slider("Number of visualizations to generate", min_value=1, max_value=10, value=2)

    # Handle dataset selection and upload
    st.sidebar.write("## Explore your data")
    st.sidebar.write("### 1. Choose a CSV or JSON file")

    datasets = [
        {"label": "Select a dataset", "url": None},
        {"label": "Cars", "url": "https://raw.githubusercontent.com/uwdata/draco/master/data/cars.csv"},
        {"label": "Weather", "url": "https://raw.githubusercontent.com/uwdata/draco/master/data/weather.json"},
    ]

    selected_dataset_label = st.sidebar.selectbox(
        'Choose a dataset',
        options=[dataset["label"] for dataset in datasets],
        index=0
    )

    upload_own_data = st.sidebar.checkbox("Upload your own data")

    if upload_own_data:
        uploaded_file = st.sidebar.file_uploader("Choose a CSV or JSON file", type=["csv", "json"])
        relevant_table_count = 0

        if uploaded_file is not None:
            # Get the original file name and extension
            file_name, file_extension = os.path.splitext(uploaded_file.name)

            # Load the data depending on the file type
            if file_extension.lower() == ".csv":
                data = pd.read_csv(uploaded_file)
            elif file_extension.lower() == ".json":
                data = pd.read_json(uploaded_file)

            # Save the data using the original file name in the data dir
            uploaded_file_path = os.path.join("data", uploaded_file.name)
            data.to_csv(uploaded_file_path, index=False)

            selected_dataset = uploaded_file_path

            datasets.append({"label": file_name, "url": uploaded_file_path})

            # st.sidebar.write("Uploaded file path: ", uploaded_file_path)
    else:
        selected_dataset = datasets[[dataset["label"]
                                     for dataset in datasets].index(selected_dataset_label)]["url"]
        relevant_table_count = 0

    #if not selected_dataset:
        #st.info("To continue, select a dataset from the sidebar on the left or upload your own.")

    st.sidebar.write("### 2. Choose from RDBMS (Customer 360)")
    with st.sidebar:
        
        personas = [
            {"label": "Select a persona", "value": None},
            # {"label": "Underwriter", "value": "Transactions"},
            {"label": "Sales Agent", "value": "Sales Agent and Policies"},
            {"label": "Customer Experience Manager", "value": "Customer Experience Manager"},
        ]

        data_summary = st.sidebar.checkbox("Add Data Summarization")
        goal_summary = st.sidebar.checkbox("Add Goal Summarization")
        data_explain = st.sidebar.checkbox("Add Data Explanation")
        own_persona = st.sidebar.checkbox("Add Your Own Persona")
        if own_persona:
            user_persona = st.sidebar.text_input("Describe your persona")
            personas.append({"label": user_persona, "value": user_persona})

        selected_persona_label = st.sidebar.selectbox(
            'Choose your persona',
            options=[persona["label"] for persona in personas],
            index=0
        )
        
        persona = personas[[persona["label"]
                                     for persona in personas].index(selected_persona_label)]["value"]
       
        if persona:
            with st.spinner("Processing your request. Please wait a moment..."):
                #start = timer()

                try:
                    #azure_deployment = "gpt-4-32k"
                    azure_deployment = "gpt-4o"
                    openai_api_version = openai_api_version
                    openai_api_key = openai_key
                    azure_endpoint = azure_endpoint

                    # driver="ODBC+Driver+17+for+SQL+Server"
                    # server="XWPF2N1VAQ"
                    # database="INCDINTHL4SDB01"
                    # username="GenAI"
                    # password="Password123"
                    # params = urllib.parse.quote_plus(
                    # 'Driver=%s;' % driver +
                    # 'Server=tcp:%s,1433;' % server +
                    # 'Database=%s;' % database +
                    # 'Uid=%s;' % username +
                    # 'Pwd={%s};' % password +
                    # 'Encrypt=yes;' +
                    # 'TrustServerCertificate=no;' +
                    # 'Connection Timeout=30;')

                    conn_str = "mssql+pyodbc://XWPF2N1VAQ/Customer360?driver=ODBC+Driver+17+for+SQL+Server&Trusted_Connection=yes"

                    schemas = []
                    schemas.append("dbo")
                    # schemas.append("esg_aggr")
                    #schemas.append("esg_process")
                    #schemas.append("esg_raw")
                    #schemas.append("esg_semantic")
                    #schemas.append("esg_user_input")
                    #schemas.append("t_esg_aggr")

                    # table_description, column_description = get_business_glossary(r"C:\Users\TM781UW\Documents\SQL Server Management Studio\Database Installation Scripts\LifePAS\LifePAS Business Glossary.xlsx")

                    # db = SQLDatabase.from_uri(conn_str, schemas=schemas, table_description=table_description, column_description=column_description, view_support=True)
                    db = SQLDatabase.from_uri(conn_str, schemas=schemas, view_support=True)
                    table_details = ""
                    for table in db._all_tables:
                        table_details += table + ", "

                    llm_32k = AzureChatOpenAI(
                        azure_deployment = azure_deployment,
                        openai_api_version = openai_api_version,
                        api_key = openai_api_key,
                        azure_endpoint = azure_endpoint,
                        temperature = 0.0,
                    )

                    table_details_prompt = """You are a SQL expert and your job is to find only relevant table names which are related to the given user                                                persona from a table list\n

                    Lets think step by step

                    1. You are given two things. A user persona and a table list.
                    2. Your task is to find relevant table names from the table list which are related to the given user persona.
                    3. The given table list consists of all table names separated by comma.
                    5. The user persona might be related to information from one or multiple tables so always read through each table in the given table                            list carefully.
                    6. STRICTLY find ONLY RELEVANT table names.
                    7. ONLY provide output as comma separated table names and nothing else.
                    8. The output would be STRICTLY a comma separated string as per below format:

                    Format:
                    <<schema1.table 1>>, <<schema1.table 2>>, <<schema2.table 3>>

                    The tables along with their descriptions are as follows:\n
                    {table_details}

                    The given user persona is - {persona}

                    Do not include introductions, explanations and conclusions.

                    Output:
                    """

                    prompt = PromptTemplate(template=table_details_prompt, input_variables=["persona", "table_details"])
                    llm_chain = LLMChain(prompt=prompt, llm=llm_32k, verbose=True)
                    # response = llm_chain.run({"persona": persona, "table_details": json.dumps(table_description)})
                    response = llm_chain.run({"persona": persona, "table_details": table_details})

                    relevant_table_names = get_relevant_tables(response)
                    print(relevant_table_names)
                    engine = create_engine(conn_str)

                    relevant_table_count = 0
                    for table in relevant_table_names:
                        sql = "select * from " + table.strip()
                        df = pd.read_sql(sql,con=engine)
                        esg_compass_relevant_datasets[table.strip()] = df
                        relevant_table_count += 1
                        
                except Exception as e:
                    st.sidebar.write("Failed to process your request. Please try again.")
                    print(e)

            #st.sidebar.write(f"Time taken: {timer() - start:.2f}s")

if relevant_table_count > 0:
    
    for table in relevant_table_names:
        # Step 3 - Generate data summary
        selected_dataset = esg_compass_relevant_datasets[table.strip()]
                
        if openai_key and selected_method:
            text_gen = llm(
                provider="openai",
                api_type="azure",
                azure_endpoint=azure_endpoint,
                api_key=openai_key,
                api_version=openai_api_version,
            )
            lida = Manager(text_gen=text_gen)
            textgen_config = TextGenerationConfig(
                n=1,
                temperature=temperature,
                model=selected_model,
                use_cache=use_cache)

            # print(selected_dataset, "\n\n", selected_method, "\n\n", textgen_config)
            st.write("## Summary")
            # **** lida.summarize *****
            summary = lida.summarize(
                selected_dataset,
                summary_method=selected_method,
                textgen_config=textgen_config)

            if data_summary:
                if "dataset_description" in summary:
                    st.write(summary["dataset_description"])

                #if "fields" in summary:
                    #fields = summary["fields"]
                    #nfields = []
                    #for field in fields:
                        #flatted_fields = {}
                        #flatted_fields["column"] = field["column"]
                        ## flatted_fields["dtype"] = field["dtype"]
                        #for row in field["properties"].keys():
                            #if row != "samples":
                                #flatted_fields[row] = field["properties"][row]
                            #else:
                                #flatted_fields[row] = str(field["properties"][row])
                        ## flatted_fields = {**flatted_fields, **field["properties"]}
                        #nfields.append(flatted_fields)
                    #nfields_df = pd.DataFrame(nfields)
                    #st.write(nfields_df)
                #else:
                    #st.write(str(summary))

            # Step 4 - Generate goals
            if summary:
                # **** lida.goals *****
                goals = lida.goals(summary, n=num_goals, textgen_config=textgen_config)
                st.write(f"## Goals ({len(goals)})")

                #default_goal = goals[0].question
                goal_questions = [goal.question for goal in goals]

                selected_goal = st.selectbox('Choose a generated goal', options=goal_questions, index=0, key="sg_sb_" + str(index))
                
                # st.markdown("### Selected Goal")
                selected_goal_index = goal_questions.index(selected_goal)
                if goal_summary:
                    st.write("**Question:** " + goals[selected_goal_index].question)
                    st.write("**Rationale:** " + goals[selected_goal_index].rationale)
                    st.write("**Visualization:** " + goals[selected_goal_index].visualization)

                selected_goal_object = goals[selected_goal_index]

                # Step 5 - Generate visualizations
                if selected_goal_object:
                    
                    # Update the visualization generation call to use the selected library.
                    st.write("## Visualizations")

                    textgen_config = TextGenerationConfig(
                        n=num_visualizations, temperature=temperature,
                        model=selected_model,
                        use_cache=use_cache)

                    # **** lida.visualize *****
                    visualizations = lida.visualize(
                        summary=summary,
                        goal=selected_goal_object,
                        textgen_config=textgen_config,
                        library=selected_library)

                    viz_titles = [f'Visualization {i+1}' for i in range(len(visualizations))]
                    selected_viz_title = st.selectbox('Choose a visualization', options=viz_titles, index=0, key="svt_sb_" + str(index))
                    selected_viz = visualizations[viz_titles.index(selected_viz_title)]

                    if selected_viz.raster:
                        from PIL import Image
                        import io
                        import base64

                        imgdata = base64.b64decode(selected_viz.raster)
                        img = Image.open(io.BytesIO(imgdata))
                        st.image(img, caption=selected_viz_title, use_column_width=True)

                    #st.write("### Visualization Code")
                    #st.code(selected_viz.code)
                    if data_explain:
                        explanations = lida.explain(code=selected_viz.code, library=selected_library, textgen_config=textgen_config) 
                        for row in explanations[0]:
                            st.write(row["section"]," ** ", row["explanation"])

        index += 1
            
    #relevant_table_count = 0
    #esg_compass_relevant_datasets = {}

else:
    # Step 3 - Generate data summary
    if openai_key and selected_dataset and selected_method:
        text_gen = llm(
            provider="openai",
            api_type="azure",
            azure_endpoint=azure_endpoint,
            api_key=openai_key,
            api_version=openai_api_version,
        )
        lida = Manager(text_gen=text_gen)
        textgen_config = TextGenerationConfig(
            n=1,
            temperature=temperature,
            model=selected_model,
            use_cache=use_cache)

        st.write("## Summary")
        # **** lida.summarize *****
        summary = lida.summarize(
            selected_dataset,
            summary_method=selected_method,
            textgen_config=textgen_config)

        if "dataset_description" in summary:
            st.write(summary["dataset_description"])

        if "fields" in summary:
            fields = summary["fields"]
            nfields = []
            for field in fields:
                flatted_fields = {}
                flatted_fields["column"] = field["column"]
                # flatted_fields["dtype"] = field["dtype"]
                for row in field["properties"].keys():
                    if row != "samples":
                        flatted_fields[row] = field["properties"][row]
                    else:
                        flatted_fields[row] = str(field["properties"][row])
                # flatted_fields = {**flatted_fields, **field["properties"]}
                nfields.append(flatted_fields)
            nfields_df = pd.DataFrame(nfields)
            st.write(nfields_df)
        else:
            st.write(str(summary))

        # Step 4 - Generate goals
        if summary:
            
            # **** lida.goals *****
            goals = lida.goals(summary, n=num_goals, textgen_config=textgen_config)
            st.write(f"## Goals ({len(goals)})")

            default_goal = goals[0].question
            goal_questions = [goal.question for goal in goals]

            #if own_goal:
                #user_goal = st.sidebar.text_input("Describe Your Goal")

                #if user_goal:

                    #new_goal = Goal(question=user_goal, visualization=user_goal, rationale="")
                    #goals.append(new_goal)
                    #goal_questions.append(new_goal.question)

            selected_goal = st.selectbox('Choose a generated goal', options=goal_questions, index=0)

            # st.markdown("### Selected Goal")
            selected_goal_index = goal_questions.index(selected_goal)
            st.write(goals[selected_goal_index])

            selected_goal_object = goals[selected_goal_index]

            # Step 5 - Generate visualizations
            if selected_goal_object:
                
                # Update the visualization generation call to use the selected library.
                st.write("## Visualizations")

                textgen_config = TextGenerationConfig(
                    n=num_visualizations, temperature=temperature,
                    model=selected_model,
                    use_cache=use_cache)

                # **** lida.visualize *****
                visualizations = lida.visualize(
                    summary=summary,
                    goal=selected_goal_object,
                    textgen_config=textgen_config,
                    library=selected_library)

                viz_titles = [f'Visualization {i+1}' for i in range(len(visualizations))]

                selected_viz_title = st.selectbox('Choose a visualization', options=viz_titles, index=0)

                selected_viz = visualizations[viz_titles.index(selected_viz_title)]

                if selected_viz.raster:
                    from PIL import Image
                    import io
                    import base64

                    imgdata = base64.b64decode(selected_viz.raster)
                    img = Image.open(io.BytesIO(imgdata))
                    st.image(img, caption=selected_viz_title, use_column_width=True)

                #st.write("### Visualization Code")
                #st.code(selected_viz.code)