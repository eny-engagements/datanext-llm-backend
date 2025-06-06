'''
Hybrid Knoledge Base (KB) for Business Glossary in Qdrant.

Following are the collections:
- Table Embeddings (reduced collection for optimization)
- Column Embeddings

Each object in the Table Embeddings collection contains:
- Schema Name
- Table Name
- Table Description
- Embedding of the Table Description

Each object in the Column Embeddings collection contains:
- Schema Name
- Table Name
- Table Description
- Column Name
- Column Description
- Embedding of the Column Description
- Column Subdomain (optional)
- PII (optional)
- Data Type (optional)
- Key (optional)

 Following uses are planned:
- Fuzzy matching of Schema Name to service user requests for Glossaries which already exist
- Keyword search for Table Name and Column Name
- Vector search of both the collectios for Map.ai

Future considerations:
- Currently, collections are created in localhost:6333. Should be changed to pipelining knowledge base to Docker container
of Qdrant Cloud on Azure.
'''

from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, PointStruct
import pandas as pd
from sentence_transformers import SentenceTransformer
import logging 

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

hf_embeddings = SentenceTransformer("./all-mpnet-base-v2/", trust_remote_code=True)

def create_qdrant_indexes(file, qdrant_host="localhost", qdrant_port=6333): # Host and port may be changed for Qdrant Cloud
    client = QdrantClient(host=qdrant_host, port=qdrant_port)
    df = pd.read_excel(file)
    df.ffill(inplace=True)

    # Schema Embeddings - Failsafe for Fuzzy Logic Matching and Sector Seggregation
    schema_df = df.groupby(['Schema Name'])[['Schema Name', 'Schema Description']].first().reset_index(drop=True)
    schema_points = []
    for idx, row in schema_df.iterrows():
        embedding = hf_embeddings.encode(str(row['Schema Description']))
        payload = {
            "Sector": str(row['Sector']),
            "Schema Name": str(row['Schema Name']),
            "Schema Description": str(row['Schema Description'])
        }
        schema_points.append(PointStruct(id=idx+1, vector=embedding.tolist(), payload=payload))
    client.recreate_collection(
        collection_name="Schema Embeddings",
        vectors_config=VectorParams(size=len(schema_points[0].vector), distance=Distance.COSINE)
    )
    client.upsert(collection_name="Schema Embeddings", points=schema_points)

    # Table Embeddings
    table_df = df.groupby(['Table Name'])[['Table Name', 'Table Description']].first().reset_index(drop=True)
    table_points = []
    for idx, row in table_df.iterrows():
        embedding = hf_embeddings.encode(str(row['Table Description']))
        payload = {
            "Schema Name": str(row['Schema Name']),
            "Table": str(row['Table Name']),
            "Table Description": str(row['Table Description'])
        }
        table_points.append(PointStruct(id=idx+1, vector=embedding.tolist(), payload=payload))

    client.recreate_collection(
        collection_name="Table Embeddings",
        vectors_config=VectorParams(size=len(table_points[0].vector), distance=Distance.COSINE)
    )
    client.upsert(collection_name="Table Embeddings", points=table_points)

    # Column Embeddings
    column_points = []
    for idx, row in df.iterrows():
        embedding = hf_embeddings.encode(str(row['Column Description']))
        payload = {
            "Schema Name": str(row['Schema Name']),
            "Table Name": str(row['Table Name']),
            "Table Description": str(row['Table Description']),
            "Column Name": str(row['Column Name']),
            "Column Description": str(row['Column Description']),
            "Data Type": row.get('Data Type', None),
            "PII": row.get('PII', None),
            "Key": row.get('Key', None)
        }
        column_points.append(PointStruct(id=idx+1, vector=embedding.tolist(), payload=payload))

    client.recreate_collection(
        collection_name="Column Embeddings",
        vectors_config=VectorParams(size=len(column_points[0].vector), distance=Distance.COSINE)
    )
    client.upsert(collection_name="Column Embeddings", points=column_points)

    
    logging.info("Qdrant collections created: 'Schema Embeddings', 'Table Embeddings' and 'Column Embeddings'")
    # return "Table Embeddings", "Column Embeddings"

