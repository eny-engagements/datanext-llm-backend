import streamlit as st
from streamlit_mermaid import st_mermaid

from typing import Optional
import os
import subprocess
import time
import pandas as pd
import numpy as np
import base64
import re
from concurrent.futures import ThreadPoolExecutor

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import AzureChatOpenAI
# from langchain_ollama.llms import OllamaLLM
# from langchain.agents import initialize_agent, AgentType
# from langchain.tools import Tool

import autogen

from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

 
gpt4o = AzureChatOpenAI(
    azure_deployment="gpt-4o",
    openai_api_version="2024-02-15-preview",
    api_key="901cce4a4b4045b39727bf1ce0e36219",
    azure_endpoint="https://aifordata-azureopenai.openai.azure.com/",
    temperature=0.5,
    max_tokens=4096
)

model = SentenceTransformer("C:/Users/TM781UW/huggingface/all-mpnet-base-v2/", trust_remote_code=True)

def load_and_validate_glossary(glossary_file):
    try:
        if glossary_file.name.endswith(".csv"):
            glossary_df = pd.read_csv(glossary_file)
        elif glossary_file.name.endswith((".xlsx", ".xls")):
            glossary_df = pd.read_excel(glossary_file)
        else:
            st.error("Invalid file format. Please upload a CSV or Excel file.")
            return None
        glossary_df['Table Description'] = glossary_df['Table Description'].replace("", np.nan)
        glossary_df['Table Description'] = glossary_df.groupby('Table Name')['Table Description'].ffill()
        glossary_df['unique_name'] = glossary_df['Table Name'].astype(str) + " - " + glossary_df['Column Name'].astype(str)
        glossary_df['full_description'] = "Table:\n" + glossary_df['Table Description'] + "\n\nColumn:\n" + glossary_df['Column Description']
        
        return glossary_df
    except Exception as e:
        st.error(f"Error reading the schema file: {e}")
        return None

def generate_embeddings(description, model):
    try:
        return model.encode(description, convert_to_tensor=True)
    except Exception as e:
        st.error(f"Error generating embeddings: {e}")
        return None

def compute_embeddings(glossary_df, model):
    futures = []
    with ThreadPoolExecutor(max_workers=20) as executor:
        for description in glossary_df['Column Description'].tolist():
            future = executor.submit(generate_embeddings, description, model)
            futures.append(future)
    return [future.result() for future in futures]

def compute_similarity_matrix(embeddings):
    try:
        similarity_matrix = np.zeros((len(embeddings), len(embeddings)))
        
        def compute_similarity(i, j):
            return cosine_similarity([embeddings[i]], [embeddings[j]])[0][0]
        
        with ThreadPoolExecutor(max_workers=50) as executor:
            results = {(i, j): executor.submit(compute_similarity, i, j) for i in range(len(embeddings)) for j in range(i+1, len(embeddings))}
        
        for (i, j), future in results.items():
            similarity_matrix[i][j] = similarity_matrix[j][i] = future.result()
        
        # st.write("Similarity matrix computed successfully!")
        return similarity_matrix
    except Exception as e:
        st.error(f"Error computing similarity matrix: {e}")
        return None

def cluster_attributes(glossary_df, similarity_matrix):
    attribute_names = glossary_df['unique_name'].tolist()
    distance_matrix = 1 - similarity_matrix
    kmeans = KMeans(n_clusters=int(len(attribute_names)/30) + 1)
    cluster_labels = kmeans.fit_predict(distance_matrix)
    
    # st.write("Clusters generated successfully!")
    # st.write(cluster_labels)
    
    cluster_df = pd.DataFrame({'entities_attributes': attribute_names, 'entities_attributes_decoded': pd.NA , 'cluster_label': cluster_labels, 'topic_cluster': pd.NA})
    return cluster_df.groupby('cluster_label')


def taggings(cluster, tags, database_desc):

    prompt_tag = PromptTemplate.from_template(
    """ 
    You are a subject-matter expert in the life insurance domain and have experience in quantitative data modelling
    with application databases which are typically used in industry.
 
    Following is a brief description of one such application database:
    {database_desc}
 
    I will now provide you with a list of names of entities and attributes from the database schema separated by hyphens.
    These have been logically grouped based on semantic similarity.
 
    Following is a list of descriptions for each entity and attribute in the cluster:
    {cluster}
 
    Your initial task is to analyse each entity and attribute description and translate them into sensible names in snake_case.
    For tagging, your primary focus should shift towards the *attribute* descriptions.
    Note, some descriptions may contain codified names and names of the application system. You will disregard these names 
    and you will focus on the description when naming it. The names of entities and attributes *should not* include
    original table names, column names or application system names.
    For tagging, your primary focus should shift towards the *attribute* descriptions.
 
    Ultimately your task is to deeply analyze the *attribute* descriptions within each group to understand their inherent 
    characteristics, purposes, and analytical relevance. Your goal is to infer the *attribute themes* or *functional themes* 
    represented by these attributes within the group. Leveraging your domain knowledge and analysis, you will provide a set 
    of tags that effectively capture:
 
    1.  The shared characteristics and analytical trends among the *attributes* in the group.
    2.  The overarching *functional theme* these attributes collectively represent.
    3.  The low-level, granular aspects of the system *as reflected by these attributes*.
 
    These tags should be designed to assist data modelers in efficiently locating relevant *attributes* when designing schemas 
    for specific analytical use cases.
 
    **Crucially, tags should not directly describe or name entities. Instead, they must provide a dimension of logical organization 
    specifically for the *attributes*, reflecting their analytical role, data characteristics, or functional purpose within the domain.**  
    Think of tags as attribute classifications or analytical categories.

    You may also use tags from the following set of tags if applicable:
    {tags}
 
    Respond in the following format:
    Tags:
    comma seperated subdomains in sentence case summarising the *attribute-level* logical relationships within the group, highlighting 
    how the *attributes* contribute to a shared functional or analytical theme.
 
    | Entity Name | Attribute Name | Tag |
    |------------|-------------|-------------|
    | [entity_name 1] | [attribute_name 1] | [tag] |
    | [entity_name 2] | [attribute_name 2] | [tag] |
    ...
    | [entity_name n] | [attribute_name n] | [tag] |
    -----------
    """
    )

    try:   
        runnable = RunnablePassthrough()
        output_parser = StrOutputParser()
        metric_chain = runnable | prompt_tag | gpt4o | output_parser
        names_tags = metric_chain.invoke({"database_desc": database_desc, "cluster": cluster, 'tags': tags})
        return names_tags
       
    except Exception as e:
        print(f"Error in tagging: {e}")
        return None

def parse_markdown_table(output):
   
    #name and subdomain into dataframe
    #extract tags
   
    lines = output.split("\n")
 
    # Extract tags before the table
    extracted_tags = set()
    # for line in lines:
    #     # if line.startswith("Tags:"):
    #     #     extracted_tags.extend([tag.strip() for tag in line.replace("Tags:", "").split(",")])
    #     if line.strip() != "" and not line.startswith("|") and line.strip().replace("Tags:", "") != "":
    #         extracted_tags.extend([tag.strip() for tag in line.strip().replace("Tags:", "").split(",")])
    #     else:
    #         break
 
    # Extract table contents
    table_data = []
    for line in lines:
        if "|" in line and not line.startswith("|----") and "Entity Name" not in line:
            columns = [col.strip() for col in line.split("|")[1:-1]]  # Remove empty first and last split
            if len(columns) == 3:  # Non-codified case
                entity_name, attribute_name, tag = columns
                extracted_tags.update(tag.strip().split(","))
            # elif len(columns) == 3:  # Codified case
            #     _, name, subdomain = columns
            else:
                continue
            table_data.append([entity_name, attribute_name, tag])
 
 
    subdomains_df = pd.DataFrame(table_data, columns=["Entity Name", "Attribute Name", "Subdomain"]) if table_data else None
 
    return subdomains_df, extracted_tags

def process_clusters(grouped, glossary_df, database_desc):
    all_tags = set()
    cluster_dfs = []
    
    for cluster_id, group in grouped:
        # st.write(f"Processing Cluster {cluster_id}...")
        filtered_df = glossary_df[glossary_df['unique_name'].isin(group['entities_attributes'].tolist())]
        response = taggings(filtered_df[['Table Description', 'Column Description']].to_string(header=True, index=False), str(all_tags), database_desc)  
        # st.write(response)
        subdomains_df, extracted_tags = parse_markdown_table(response)
        # st.write(extracted_tags)
        
        all_tags.update(extracted_tags)
        
        if subdomains_df is not None:
            cluster_dfs.append(subdomains_df)
    
    combined_cluster_df = pd.concat(cluster_dfs, ignore_index=True) if cluster_dfs else None
    
    # st.write("Final DataFrame:", combined_cluster_df)
    # st.write("Cluster Tag Sets:", all_tags)
    
    return combined_cluster_df, all_tags

#data modelling
# def decompose(user_query):
#     try:
#         decompose_prompt = PromptTemplate.from_template(
# """
# You are an expert in quantitative modeling and business intelligence reporting, with a focus on the life insurance
# domain. Your life insurance organization wants to perform data warehousing in a medallion architecture to service 
# regulatory reporting and analytics across multiple use cases. This requires a very robust silver layer with comprehensive
# base level attribution.

# Following is one such analytical use case:
# {user_query}

# Your task is to understand the use case, determine the measures, indicators and statistical calculations that are relevant
# to the use case and consequently the base level attributions necessary for modelling the silver layer. You will finally 
# explain how these are summarised alongwith a business centric explanation of the KPIs which should inform a technical and
# non-technical in writing the DDL of the silver layer schema.

# Begin your response with explanation of the analytical case and advantages of the set of measures, indicators and attributions
# you will be mentioning, then complete the response in the following format:
# ### Base Measures:\n
# [bulleted list of measures]\n\n
# ### Derived Measures:\n
# - **[measure 1]**:\n 
#     [purpose of aggregation]\n
#     [formula]\n
# - **[measure 2]**:\n 
#     [purpose of aggregation]\n 
#     [formula]\n
# ....
# ### KPIs with glossary and formulae:\n
# - **[indicator 1]**:\n 
#     [business centric explanation of indicator]\n
#     [formula]\n
# - **[indicator 2]**:\n 
#     [business centric explanation of indicator]\n
#     [formulae]\n
# ....
# """
# )
#         runnable = RunnablePassthrough()
#         output_parser = StrOutputParser()
#         metric_chain = runnable | decompose_prompt | gpt4o | output_parser

    
# def agents(use_case_query, combined_cluster_df, all_tags):
#     def search_attribute_tags_df(user_query):
#         """Searches the entity_attribute_tags_df for rows matching the search tags."""
#         result_list = []
#         try:
#             for tag in user_query.split(", "):
#                 result = combined_cluster_df[combined_cluster_df['Subdomain'].str.lower().str.contains(tag.lower())][['Entity Name', 'Attribute Name', 'Subdomain']]
#                 if not result.empty:
#                     result_list.append(result)
            
#             if len(result_list) > 0:
#                 return pd.concat(result_list, ignore_index=True)
#             else:
#                 return "No matching attributes found for the query."
#         except Exception as e:
#             return "Encountered an rror during dataframe search: {e}"
    
#     try:
#         attribute_search_tool = Tool(
#             name="attribute_tag_search",
#             func=search_attribute_tags_df,
#             description=f"""
#             Useful for searching for attributes and entities based on tags. Input should be a comma separated list of tags or keywords to 
#             search for in the tags column. Only use search tags from the following set: 
#             {str(all_tags).strip("{}")}. 
#             Returns a string of matching entity and attribute names along with their tags.
#             """
#         )
 
#         # Optimized for Tag-Based Search and Analysis
#         attribute_agent_prompt_template = PromptTemplate.from_template(
#         """
#         You are a rework oriented agent, a highly specialized expert in quickly searching and analyzing **life insurance attribute metadata** based on 
#         tags.
#         You are tasked by the schema designer agent to find relevant **life insurance entities and attributes** from a corporate database based on a specific request.
#         Your primary goal is to efficiently search this database using the tags provided by the schema designer agent and return a concise set of the **most 
#         relevant life insurance entity-attribute pairs**.
        
#         Use the 'attribute_tag_search' tool to search the life insurance data.
                
#         Instructions:
#         1. Understand the task from the schema designer agent.
#         2. Use the 'attribute_tag_search' tool with the relevant tags to search for attributes.
#         3. Analyze the search results to identify the most relevant **life insurance entity and attribute names** that align with the task. 
#         Focus on extracting entity-attribute pairs and a brief justification of their relevance within the context of the task.
#         4. Return the relevant **life insurance entity-attribute pairs** and explain why they are relevant based on the context of the task.
#         If no relevant attributes are found, explicitly state "No relevant life insurance attributes found for the task."
                
#         Begin!
        
#         ReactAgent Task: {task}
#         """
#         )
 
#         attribute_tools = [attribute_search_tool]
 
#         attribute_agent = initialize_agent(
#             agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
#             # agent=AgentType.SELF_ASK_WITH_SEARCH,
#             llm=gpt4o,
#             tools=attribute_tools,
#             verbose=True,
#             max_iterations=3, # Limit iterations for RewooAgent
#             early_stopping_method="generate",
#             agent_kwargs={
#                 "prompt": attribute_agent_prompt_template, 
#                 "handle_parsing_errors": True
#             }
#         )
        
#         # Optimized for Decomposition, Schema Generation and Orchestration
#         react_agent_prompt_template = PromptTemplate.from_template(
#         """
#         You are a schema designer agent capable of reflection and action. You are an expert in data engineering and quantitative
#         modelling, specifically for **life insurance domain**.
#         You are requested by a user to create a complete schema for a use case related to life insurance.
#         Your task is to respond with a robust and complete schema of a bronze layer data model containing base level attributions
#         properly organized within appropriate entities, data types and logical key constraints designed to be aggregated
#         into a gold layer and service the use case.
#         You will leverage your deep understanding of life insurance concepts in drafting the schema and augment it with attributes 
#         available across application databases by using the attribute_agent as needed.
        
#         You have access to the attribute_agent to be used multiple times to find relevant entities and attributes based on the tasks 
#         you assign it.
        
#         Tools:
#         1. attribute_agent:  Specifically useful for finding relevant **life insurance entities and attributes**
#         from a corporate database based on tags. Input should be task descriptions focused on augmenting your .
        
#         Instructions:
#         1. **Use Case:** - Understand the use case in the input.
#         2. **Decomposition:** - Decompose the use case in the input to augment your 
#         understanding of the analytical requirements, focusing on life insurance concepts, objects, entities, metrics and indicators, 
#         based on which you can draft a detailed step by step plan to design the schema of the data model.
#         3. **Draft a schema:** Based on your decomposition, elaborate various aspects of the logical data model. Think about core life 
#         insurance concepts and objects. Create the initial structure of the data model by linking entities logically. Create a step-by-step plan. 
#         This plan will inform you in tasking the attribute agent with discovering various attributes so that you can use them to design a 
#         comprehensive schema.
#         4. **Augment with application database attributes using attribute_agent:**
#             - *Do not* assign all attribute search tasks at once. Break it down into multiple granular tasks.
#             - Formulate precise tasks for the attribute_agent to search for relevant attributes.
#             - Delegate multiple such tasks to the 'attribute_agent', ensuring the task descriptions are clearly focused on your requirements at
#             each stage of the design lifecycle. The task descriptions should also include your progress in designing the schema at the current stage.
#             - Incorporate the **entities and attributes** into your schema logically.
#         5. **Refine and Finalize Schema:** Review the augmented schema. Ensure it is comprehensive, robust, and directly addresses the use case.
#         Pay special attention to relationships that are crucial in life insurance data, such as policy-client, policy-claim, etc.
#         6. **Output the Final Data Schema:** Present the final data schema in a clear and structured format (e.g., list of
#         entities with their attributes, key constraints), optimized for OLAP.
        
        
#         Begin!
        
#         Use Case: {use_case_query}
#         """
#         )
 

#         react_tools = [
#             Tool(
#                 name="attribute_agent",
#                 func=attribute_agent.run,
#                 description=""" 
#                 Useful for finding relevant entities and attributes from a corporate database based on tags. Input should be a task for the agent.
#                 """
#             )
#         ]
 
#         react_agent = initialize_agent(
#             tools=react_tools,
#             llm=gpt4o,
#             agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
#             # agent=AgentType.SELF_ASK_WITH_SEARCH,
#             verbose=True,
#             max_iterations=5, # Limit overall ReactAgent iterations
#             agent_kwargs={
#                 "prompt": react_agent_prompt_template,
#                 "handle_parsing_errors": True
#             },
#             early_stopping_method="generate"
#         )
 
#         schema_output = react_agent.run({"input": {"use_case_query": use_case_query}})

#         return schema_output
#     except Exception as e:
#         st.error(f"Error initializing Agents: {e}")
#         return None

def get_data_model_or_ontology_images():
    try:
        return ', '.join(os.listdir("./input/input to modeller/data models"))
    except Exception as e:
        return f"Error listing images"
    
def agents(use_case_query, combined_cluster_df, all_tags):
    """
    Implements agents for schema design using AutoGen, optimized for life insurance domain.

    Args:
        use_case_query (str): The user's use case query.
        combined_cluster_df (pd.DataFrame): DataFrame containing entity-attribute metadata.
        all_tags (set): Set of all available tags.

    Returns:
        str: The generated schema output.
    """

    config_list = [ 
        { 
            "model": "gpt-4o", 
            "api_key": "901cce4a4b4045b39727bf1ce0e36219", 
            "base_url": "https://aifordata-azureopenai.openai.azure.com/", 
            "api_type": "azure", 
            "api_version": "2024-02-15-preview",
            "max_tokens": 4096,
            "temperature": 0.5
        } 
    ]

    llm_config = {"config_list": config_list, "seed": 42}

    # User Proxy Agent
    user_proxy = autogen.UserProxyAgent(
        name="user_proxy",
        human_input_mode="NEVER",
        # max_consecutive_auto_reply=15, 
        code_execution_config=False,
        is_termination_msg=lambda msg: isinstance(msg, dict) and msg.get("content") and "DONE" in msg["content"] 
    )

    def search_attribute_tags_df(user_query: str) -> str:
        """Searches a dataframe containing entities and attributes tagged by subdomain, for rows matching the search tags."""
        result_list = []
        try:
            for tag in user_query.split(","):
                # result = combined_cluster_df[combined_cluster_df['Subdomain'].str.lower().str.contains(tag.strip().lower())][['Entity Name', 'Attribute Name', 'Subdomain']]
                # result = combined_cluster_df[tag.strip().lower() in combined_cluster_df['Subdomain'].str.split(", ")][['Entity Name', 'Attribute Name', 'Subdomain']]
                result = combined_cluster_df[combined_cluster_df['Subdomain'].str.split(", ").apply(lambda x: tag.strip().lower() in [i.lower() for i in x])][['Attribute Name', 'Subdomain']]
                if not result.empty:
                    result_list.append(result)

            if len(result_list) > 0:
                return pd.concat(result_list, ignore_index=True).to_string()
            else:
                return "No matching attributes found for the query."
        except:
            return "Encountered an error during dataframe search"

    # Attribute Search Agent
    attribute_agent = autogen.AssistantAgent(
        name="attribute_agent",
        llm_config=llm_config,
        system_message=f"""
        You are a rework oriented agent, a highly specialized expert in quickly searching and analyzing **operation database metadata** based on 
        tags.
        You are tasked by the schema designer agent to find relevant **attributes** from operational databases based on a specific request.
        Your primary goal is to efficiently search this database using the tags provided by the schema designer agent and return the set of the 
        **relevant attributes**.
        
        Use the 'search_attribute_tags_df' function to search the data using the following tags only.
        Available Tags: {str(all_tags).strip("{}")}
                
        Instructions:
        1. Understand the task from the schema designer agent.
        2. Use the 'search_attribute_tags_df' function with the relevant tags out of the available tags to search for attributes.
        3. Analyze the search results to identify **attributes** that may be modelled across dimensions and facts related to the task. 
        4. Return the attributes you have determined to align with the task (no descriptions required), a classification of whether an
        attribute should be in a fact or dimension entity by putting "fact" or "dimension" next to it and a classification of whether a 
        particular attribute contains PII (for e.g., contact information, Aadhar Number, PAN, etc) by putting "pii" next to the attribute.
        5. Once you are done, end your response with "BYE". If no matching attributes are found for the search query or no relevant attributes 
        are found for the task from the search results, explicitly state "No relevant attributes found for the task. BYE".
        """,
        # is_termination_msg=lambda msg: 'BYE' in msg["content"]
    )
    attribute_agent.register_for_llm(
        name="search_attribute_tags_df", 
        description="""
        Useful for searching for attributes and entities based on tags. Input should be a comma separated list of tags or keywords to 
        search for in the tags column. Returns a string of matching entity and attribute names along with their tags.
        """,
        api_style="function"
    )(search_attribute_tags_df)

    def retrieve_data_model_or_ontology_image(filename: str) -> str:
        "Retrieves the image from the given filename and returns a string of the base 64 encoding of the image"
        try:
            with open(f"./input/input to modeller/data models/{filename}", "rb") as file:
                return base64.b64encode(file.read()).decode("utf-8")
        except Exception as e:
            return f"Error retrieving image: {e}"
    
    model_analysis_agent = autogen.AssistantAgent(
        name="model_analysis_agent",
        llm_config=llm_config,
        system_message=f"""
        You are an expert agent in data science, data modelling and data engineering in multiple industries. You are approached by the 
        schema_designer_agent with a use case for a data model and some metrics and indicators. You may optionally be provided with
        an incomplete schema. You may also be provided with image data of some relevant standard data model or ontology, that will inform you in 
        industry best practices for data modelling and expectations in terms of structure and data points contained within a schema. 
        Your goal is to provide technical feedback, suggestions and insights to the schema_designer_agent, which it will use to design an 
        industry grade data model.

        If you are provided with an entity relationship diagram of a standard data model, you will analyse the image of 
        the entity relationship diagram and assess the design of the standard data model with respect to the following aspects:
        - Granularity of fact data
        - Key dimensions and their purposes in the context of the use case
        - Slowly changing dimensions
        - Type of fact tables (transactional, periodic)
        - Data integrity constraints
        - Separations among entities and the rationale behind the separations in terms of security of sensitive attributes, access control, 
        data volume and performance for read and update (a general rule is that PII should be contained in a separate entity)

        If you are provided with an ontology for a certain domain, you will analyse the image of the ontology and extract insights
        pertaining to entities, concepts, links among them and aggregations. Extract metrics and indicators from the ontology if a semantic
        ontology has been provided. Use them as an inspiration and suggest metrics and key performance indicators as per the analytical 
        requirements specified in the use case. This decomposition does not need to be exhaustive. 

        If you are provided with a schema, it may not be complete. You will extract insights about its design principles. You will use your analysis 
        of the entity relationship diagram or ontology to reason about how to refine and extend the schema. This reasoning should serve as an inspiration
        for the schema_designer_agent.

        If you are provided with a use case accompanied by a set of KPIs, you will determine every potential dimension and fact (granular and aggregated), 
        semantics, slowly changing dimensions, granularity of facts required, schema architecture and performance and security considerations for the data model that 
        should support the analytics for that particular industry.

        Ultimately, your analysis will help you understand the design principles for a data model that should be adhered to when designing a schema 
        for the use case. Based on your understanding, you will provide technical feedback, suggestions and insights to the schema_designer_agent.
        Please note that the schema_designer_agent does not have access to the image itself, so your communications should should not use identifying 
        references (for e.g, specific naming conventions, numberings, etc) to the image. Do not suggest draft schemas. The responsibility of creating 
        schemas is reserved to the schema_designer_agent, while your role is that of a mentor. End your response with "BYE". 
        """,
        # is_termination_msg=lambda msg: 'BYE' in msg["content"]
    )

    def task_attribute_agent(task_description: str) -> str:
        """Delegates the task to the attribute_agent and returns its analysis"""
        try:
            user_proxy.initiate_chat(
                attribute_agent,
                message=task_description,
                clear_history=False,
                max_turns=2,
                summary_method="last_msg"
            )
        except:
            return "Search failed."
        return attribute_agent.last_message()["content"]

    def task_model_analysis_agent(task_description: str, filename: Optional[str]) -> str:
        """Requests feedback from the model_analysis_agent and returns its analysis"""
        try:
            if filename is not None:
                image_data = retrieve_data_model_or_ontology_image(filename)
                message = {
                    "content": [
                        {
                            "type": "text",
                            "text": task_description
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64, {image_data}"
                            }
                        }
                    ]
                }
            else:
                message = task_description

            user_proxy.initiate_chat(
                model_analysis_agent,
                message=message,
                clear_history=False,
                max_turns=1,
                summary_method="last_msg"
            )

        except Exception as e:
            return str(e)
        return model_analysis_agent.last_message()["content"]

    all_image_files = get_data_model_or_ontology_images()

    # Schema Designer Agent
    schema_designer_agent = autogen.AssistantAgent(
        name="schema_designer_agent",
        human_input_mode="NEVER",
        llm_config=llm_config,
        system_message=f"""
        You are a schema designer agent capable of reflection and action, with expertise in data science, data modelling and data engineering 
        across multiple industries. 
        You are requested by a user to create a complete schema for a use case related to a specified industry.
        Your task is to respond with a robust and complete schema of a data model, modelling all the relevant attributes,
        properly organized within appropriate entities, accompanied by accurate data types and logical key constraints designed to be aggregated
        into a semantic layer which will service the use case.
        You will leverage your deep subject matter expertise of various industries in designing the schema and augment it with attributes 
        available across operational databases by using the attribute_agent as needed.
        
        You have access to the following functions:
        - task_attribute_agent: a function to be used multiple times to delegate tasks to the attribute_agent to 
        find relevant entities and attributes based on the task description you pass it. 
        - task_model_analysis_agent: a function for receiving feedback from the model_analysis_agent, based on the analysis 
        and decomposition of the use case and a data model, if provided in the use case, which you will pass it within the task_description 
        parameter. You will pass an image filename in the filename parameter or None if no suitable images are available.

        Instructions:

        1. **Understand the Use Case:** - Reason about the intent and technical aspects of the use case in the input.

        2. **Suggestions from model_analysis_agent**: You will pass the complete use case as well as your analysis of the use case obtained in step 1 to the model_analysis_agent. 
        You will also pass a filename from the following list of image filenames:
        {all_image_files}
        You will determine the filename to use for calling the function by semantically understanding the filenames as per the context of the use case.
        This image file is either an entity relationship diagram of enterprise data model or diagram of ontology for a domain. This image file is useful
        for a comparative analysis and for informing the model_analysis_agent in industry best practices for data modelling.
        If none of the listed images are highly relevant to the use case and associated industry, you may specify filename as None (for e.g., if the image filenames listed point to the fact
        that it may be relevant to life insurance, however the use case is for health insurance, you will not use that image). If a schema or a set of KPIs of interest in the analytical use
        case has been provided in the use case, you will include them in their entirety within the task_description.

        3. **Create a plan:** Only after receiving suggestions and insights from model_analysis_agent in step 2, you will analyse it and create a step-by-step plan. 
        This plan will inform you in tasking the attribute_agent with exploring various attributes so that you can use them to design a comprehensive schema. 

        4. **Explore source system attributes using attribute_agent:**
            - Attributes from the following subdomains are available to you:
            Available Tags: {str(all_tags).strip("{}")}
            - *Do not* assign all attribute search tasks at once. Break it down into multiple granular tasks.
            - If attributes from certain domains are provided in the use case, the tasks should elaborate on them so that attribute_agent is aware of
            how to provide attributes that are absent in the use case.
            - Formulate natural language tasks for the attribute_agent to search for potential attributes which need to be modelled across relevant dimensions 
            and facts (granular and aggregated) with respect to the listed subdomains. 
            - Delegate multiple such tasks to the 'attribute_agent', ensuring the task descriptions clearly specify your requirements at
            each stage of the design lifecycle.

        5. **Design and Finalise Schema:** After all tasks in step 4 is completed, analyse the **entities and attributes** to determine a logical organization for them 
        i.e. whether an attribute should be in a fact or dimension table. You will design the schema with respect to the suggestion from model_analysis_agent obtained in Step 2. 
        Unless denormalisation is specified in the use case, ensure that attributes other than foreign key constraints are not duplicated.
        Ensure the schema is complete, comprehensive and directly addresses the use case. Present the DDL of the final schema in a clear and structured format as follows:
        Final Schema:
        (SQL DDL in the dialect specified in use case)

        If the DDL can not be presented in a single message, you will end with "CONTINUED" only and carryover to the next message. You will do this till the entire schema
        has been presented. You will ensure that the complete DDL is presented. Finally, you will end your response with "DONE". Do not include introductions, explanations
        and conclusions in the final DDL.
        """,
        # is_termination_msg=lambda msg: "DONE" in msg["content"]
    )
    schema_designer_agent.register_for_llm(
        name="task_model_analysis_agent", 
        description="""
        A function to request feedback on the draft schema from the model_analysis_agent. Input should be the following two parameters:
        task_description: a string containing use case, decomposition of the analytical requirements and a schema, if provided in the use case.
        filename: a string containing the filename of the relevant image file.
        Returns a string of the feedback and meaningful insights.
        """,
        api_style="function"
    )(task_model_analysis_agent)

    schema_designer_agent.register_for_llm(
        name="task_attribute_agent", 
        description="""
        A function to delegate attribute search and analysis tasks to the attribute_agent. Input should be a task description string.
        Returns a string of the relevant entities and attributes according to the task.
        """,
        api_style="function"
    )(task_attribute_agent)

    user_proxy.register_function(
        function_map={
            "task_model_analysis_agent": task_model_analysis_agent,
            "task_attribute_agent": task_attribute_agent,
            "search_attribute_tags_df": search_attribute_tags_df
            # "retrieve_data_model_or_ontology_image": retrieve_data_model_or_ontology_image
        }
    )

    # Start the conversation
    user_proxy.initiate_chat(
        schema_designer_agent,
        message=f"Use Case: {use_case_query}",
        summary_method="last_msg"
    )

    # Extract the final response
    # schema_output = schema_designer_agent.last_message()["content"]
    schema_designer_chat = schema_designer_agent.chat_messages.get(user_proxy)
    schema_output = "\n".join([
        msg["content"] for msg in list(schema_designer_chat) 
        # if isinstance(msg, dict) 
        # if msg["role"] == "schema_designer_agent"
        if ( 'CONTINUED' in str(msg["content"])
        or 'DONE' in str(msg["content"]) )
    ])
    # schema_output = schema_output[schema_output.find("Final Schema") + 13:]
    schema_output = schema_output[schema_output.find("```sql"):schema_output.rfind("```")]
    schema_output = schema_output.replace("DONE", "").replace("CONTINUED", "").replace("```sql", "").replace("```", "")
    print(schema_output)

    return schema_output

def generate_ddl(use_case_query, schema_output):
    ddl_prompt = PromptTemplate.from_template(
    """ 
    You are an expert in writing SQL Queries. Your task is to write a DDL for the following schema:
    {schema}

    Respond only with the DDL such that it can serve as a complete Physical Data Model (PDM) and end your response with "----".
    If you can not complete the DDL for the entire schema, I will ask you to continue from where you left off. 
    Do not include introductions, explanations or conclusions. 
    """
    )

    ddl_continue_prompt = PromptTemplate.from_template(
    """ 
    You are an expert in writing SQL Queries. Your task is to write a DDL for the following schema:
    {schema}

    This is the DDL you have generated so far:
    {ddl_code}

    Respond only with the remaining DDL end your response with "----". Do not include introductions, explanations or conclusions.
    """
    )

    runnable = RunnablePassthrough()
    output_parser = StrOutputParser()
    ddl_chain = runnable | ddl_prompt | gpt4o | output_parser
    ddl_continue_chain = runnable | ddl_continue_prompt | gpt4o | output_parser

    ddl_code = ddl_chain.invoke({"schema": schema_output})
    ddl_code = ddl_code.replace("```sql", "").replace("```", "")

    while not ddl_code.endswith("----"):
        if not ddl_code.endswith(");"):
            ddl_code = "\n".join(ddl_code.split("\n")[:-1])
        temp = ddl_continue_chain.invoke({"schema": schema_output, "ddl_code": ddl_code})
        temp = temp.replace("```sql", "").replace("```", "")
        ddl_code += "\n" + temp

    return ddl_code

def generate_mermaid_code(use_case_query, schema_output):
    prompt_mermaid = PromptTemplate(
        input_variables=["entities_response"],
        template="""
            You are an expert in Mermaid.js ER diagram generation with a deep understanding of entity-relationship modeling.
            Using the SQL DDL of a schema created by a data modeller:
            {entities_response},

            generate Mermaid.js code for an ER diagram of the schema .
       
            ONLY give the Mermaid.js code for the ER diagram, do not include any other information like introduction or explanation.
            You will provide an appropriate interaction label between two entities, clearly explaining how two entities interact with each other in business operations.
            For example, if a customer places an order, the label for interaction between customer and order would be "places".
            If an order contains a line-item, the label for interaction between order and line-item would be "contains".
            Avoid using "reference" and "associate" in interaction labels.
            If there are multiple layers in the DDL (for e.g., bronze layer, silver layer, gold layer), the diagram should present them distinguishably.
            You may redact attributes in the diagram to upto 5 attributes since the purpose of the diagram is to serve as a Logical Data Model (LDM). 
            Only in cases where an entity contains more than 5 attributes, entities must properly indicate that attributes have been redacted with "misc other_attributes".
            You should not use "misc other_attributes" if there are 5 or fewer attributes in a certain entity.
            Follow Mermaid.js syntax strictly. Be careful not to repeat the relationship syntax, like ||--o{{--o{{, as it will cause an error.
 
            Format:
            erDiagram
                Entity1 ||--o{{ Entity2 : "(appropriate interaction label)"
                Entity1 {{
                    type attribute1
                    type attribute2
                    type attribute3
                    type attribute4
                    type attribute5
                    misc other_attributes
                }}
                Entity2 {{
                    type attribute1
                    type attribute2
                    type attribute3
                }}     

        """
    )
   
 
    try:
        
        # Format the prompt
        # input_data = {"entities_response": schema_output}
        # prompt_runner = RunnableLambda(lambda x: prompt_mermaid.format(entities_response=x["entities_response"]))
        runnable = RunnablePassthrough()
        output_parser = StrOutputParser()
        # mermaid_chain = prompt_runner | gpt4o | output_parser
        mermaid_chain = runnable | prompt_mermaid | gpt4o | output_parser
   
        # Invoke GPT-4o
        # mermaid_code = mermaid_chain.invoke(input_data)
        mermaid_code = mermaid_chain.invoke({"entities_response": schema_output, "use_case": use_case_query})
        mermaid_code = mermaid_code.replace("```mermaid", "").replace("```", "")

 
        # Validate Mermaid.js syntax
        # validate_mermaid_syntax(mermaid_code)
 
    except ValueError as ve:
        st.error(f"Error parsing Mermaid.js code: {ve}")
        return None
    except Exception as e:
        st.error(f"Error in generating Mermaid.js code: {e}")
        return None
 
    return mermaid_code
 
def validate_mermaid_syntax(mermaid_code):
    try:
        if not mermaid_code.startswith("erDiagram"):
            raise ValueError("Mermaid.js code must start with 'erDiagram'.")
 
        # Regex to match valid relationships
        relationship_pattern = r'\w+ \|\|--o\{ \w+ : "[^"]+"'
        if not re.search(relationship_pattern, mermaid_code):
            raise ValueError("Mermaid.js code does not contain valid relationships.")
       
        attribute_pattern = r'\{ + \w + \}'
        if not re.search(attribute_pattern):
            raise ValueError("Mermaid code does not have valid syntax of attributes. Please regenerate the code.")
 
        st.success("Mermaid.js syntax validation passed!")
        return True
    except ValueError as e:
        st.error(f"Mermaid.js syntax validation failed: {e}")
        return False
    except Exception as e:
        st.error(f"Error in validating Mermaid.js syntax: {e}")
        return False
 
def format_er_diagram(mermaid_code):
    def replace_with_hyphens(match):
        return match.group(0).replace(" ", "-")
 
    # Match entity names and relationships (before `{` or in `||--o{` lines), but not the description after `:`
    mermaid_code = re.sub(r"(?<=erDiagram\s*\n)(.*?)(?=\s*\{)|(\w[\w ]+\w)(?=\s*\|\|--o{)", replace_with_hyphens, mermaid_code, flags=re.MULTILINE)
 
    # Match attribute names inside entity definitions (not data types)
    mermaid_code = re.sub(r"(?<=\n\s{4}string\s)(\w[\w ]+\w)", replace_with_hyphens, mermaid_code)
 
    return mermaid_code

def generate_er_diagram(mermaid_code):
#display the er diagram
# Example ER diagram input from LL
 
    #formatted_mermaid_code = format_er_diagram(mermaid_code)
    #print(formatted_mermaid_code)
    # st.subheader("Data Model")
    #st_mermaid(formatted_mermaid_code)
    er_diagram = st_mermaid(mermaid_code, key="er_diagram1", pan=False, zoom=False)
    return er_diagram

# Function to convert Mermaid code to SVG
def generate_svg(mermaid_code):
    mermaid_file = "./output/modeller/diagram.mmd"
    svg_file = "diagram.svg"
    MMDC_PATH = r"C:\Users\TM781UW\AppData\Roaming\npm\mmdc.cmd"

    with open(mermaid_file, "w") as f:
        f.write(mermaid_code)  # Save Mermaid code to a file

    # Run Mermaid CLI with full path
    subprocess.run([MMDC_PATH, "-i", mermaid_file, "-o", svg_file], check=True)

    # Read the generated SVG content
    with open(svg_file, "r") as f:
        svg_content = f.read()

    if os.path.exists(mermaid_file):
        os.remove(mermaid_file)
    if os.path.exists(svg_file):
        os.remove(svg_file)

    return svg_content

# Function to create a download link for SVG
def get_svg_download_link(svg_content):
    b64 = base64.b64encode(svg_content.encode()).decode()
    href = f'<a href="data:image/svg+xml;base64,{b64}" download="er_diagram.svg">Download SVG</a>'
    return href


# Streamlit
if "user_query" not in st.session_state:
    st.session_state["query"] = ""
if 'mermaid_code' not in st.session_state:
    st.session_state['mermaid_code'] = None
if 'ddl_code' not in st.session_state:
    st.session_state['ddl_code'] = None
if 'er_diagram' not in st.session_state:
    st.session_state['er_diagram'] = None
if 'glossary_df' not in st.session_state:
    st.session_state['glossary_df'] = None
if 'embeddings' not in st.session_state:
    st.session_state['embeddings'] = None
if 'similarity_matrix' not in st.session_state:
    st.session_state['similarity_matrix'] = None
if 'cluster_groups' not in st.session_state:
    st.session_state['cluster_groups'] = None
if 'combined_cluster_df' not in st.session_state:
    st.session_state['combined_cluster_df'] = None
if 'all_tags' not in st.session_state:
    st.session_state['all_tags'] = None
if 'use_case_query' not in st.session_state:
    st.session_state['use_case_query'] = None
if 'entities_response' not in st.session_state:
    st.session_state['entities_response'] = None

st.set_page_config(
    layout="wide",
    page_title="Create Data Models",
    page_icon="🧬",
)

st.subheader("Create Data Models for Various Use Cases with AI 🧬")
st.markdown(
"""
This application can quickly create data models for any use case by carefully analysing source system attributions in various 
industry formats.

----
""")

st.sidebar.header("Source System Attribution")
glossary_file = st.sidebar.file_uploader("Upload Source System Glossary (CSV, XLSX, XLS)", type=["csv", "xlsx", "xls"])
database_desc = st.sidebar.text_area("Enter Source System Description")
st.sidebar.write("### OR")
combined_cluster_file = st.sidebar.file_uploader("Upload Collated Source System Attributes", type=["xlsx"])

st.subheader("Use Case")
user_query = st.text_area("Enter Your Query")

if (glossary_file and database_desc.strip() != "") or combined_cluster_file:
    if glossary_file:
        st.session_state['glossary_df'] = load_and_validate_glossary(glossary_file)
    if combined_cluster_file:
        st.session_state['combined_cluster_df'] = pd.read_excel(combined_cluster_file)
        st.session_state['all_tags'] = st.session_state['combined_cluster_df']['Subdomain'].unique()
    if user_query.strip() != "":
        if st.button("Create Data Model"):
            if st.session_state['glossary_df'] is not None and st.session_state['combined_cluster_df'] is None:
                with st.spinner("Collating Source System Attribution..."):
                    st.session_state['embeddings'] = compute_embeddings(st.session_state['glossary_df'], model)
                
                    if st.session_state['embeddings'] is not None:
                        st.session_state['similarity_matrix'] = compute_similarity_matrix(st.session_state['embeddings'])                
                        cluster_groups = cluster_attributes(st.session_state['glossary_df'], st.session_state['similarity_matrix'])                
                        combined_cluster_df, all_tags = process_clusters(cluster_groups, st.session_state['glossary_df'], database_desc)

                        st.session_state['combined_cluster_df'] = combined_cluster_df
                        st.session_state['combined_cluster_df'].to_excel(f"./output/modeller/{glossary_file.name.replace(".xlsx", "")} collated.xlsx", index=False)
                        st.session_state['all_tags'] = all_tags
                    
            if st.session_state['combined_cluster_df'] is not None and st.session_state['all_tags'] is not None:
                # with st.spinner("Collating Source System Attribution..."):
                #     time.sleep(3)
                with st.spinner("Creating Data Model Schema..."):
                    # try:
                        st.session_state['entities_response'] = agents(user_query, st.session_state['combined_cluster_df'], st.session_state['all_tags'])
                    # except Exception as e:
                    #     print(e)
                    #     pass
                    
            if st.session_state['entities_response'] is not None:
                with st.spinner("Generating ER Diagam of Data Model..."):
                    st.session_state['mermaid_code'] = generate_mermaid_code(user_query, st.session_state['entities_response'])
                # with st.spinner("Generating DDL for the Schema..."):
                #     st.session_state['ddl_code'] = generate_ddl(user_query, st.session_state['entities_response'])

if st.session_state['mermaid_code'] is not None:
    st.subheader("Entity Relationship Diagram")
    generate_er_diagram(st.session_state['mermaid_code'])

    if st.button("Save ER Diagram"):
        try:
            svg_content = generate_svg(st.session_state['mermaid_code'])
            st.markdown(get_svg_download_link(svg_content), unsafe_allow_html=True)
            st.success("SVG file is ready for download!")
        except Exception as e:
            st.error(f"Error generating SVG: {e}")

# if st.session_state['ddl_code'] is not None:
    st.subheader("Schema DDL")
    # st.code(st.session_state['ddl_code'])
    st.code(st.session_state['entities_response'])
    if st.button("Save DDL"):
        try:
            with open("./output/modeller/DDL.txt", "w") as file:
                file.write(st.session_state['ddl_code'])
            st.success("DDL saved to ./output/modeller")
        except Exception as e:
            st.error("Could not save DDL")


                    
        