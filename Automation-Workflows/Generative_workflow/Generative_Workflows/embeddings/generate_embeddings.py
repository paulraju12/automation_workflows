import json
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from dotenv import load_dotenv
import os

load_dotenv()

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
pc=Pinecone(api_key=os.getenv('PINECONE_API_KEY'), environment=os.getenv('PINECONE_ENV'))
index_name = 'workflow-components'

if index_name not in pc.list_indexes().names():
    pc.delete_index(index_name)
    pc.create_index(
        name=index_name,
        dimension=384,  # MiniLM-L6 outputs 384-dim vectors
        metric='cosine',  # Common metric for text embeddings
        spec={'serverless': {'cloud': 'aws', 'region': os.getenv('PINECONE_ENV')}}
    )

# Connect to the index
index = pc.Index(index_name)

with open('./data/connectorSnippets.json', 'r') as f:
    connectors = json.load(f)['data']['viewer']['connectorSnippets']['edges']
with open('./data/scm_providers.json', 'r') as f:
    providers = json.load(f)['data']
# print(f"Loaded {len(connectors)} connectors: {[c['node']['name'] for c in connectors]}")
# print(f"Loaded {len(providers)} providers: {[p['name'] for p in providers]}")

connector_texts = [json.dumps(c['node']) for c in connectors]
provider_texts = [json.dumps(p) for p in providers]

connector_embeddings = model.encode(connector_texts)
provider_embeddings = model.encode(provider_texts)

connector_vectors = [(c['node']['name'], emb.tolist(), {
    'type': 'connector',
    'name': c['node']['name'],
    'data': json.dumps(c['node']['data']),
    'structure': json.dumps(c['node']['structure'])
}) for c, emb in zip(connectors, connector_embeddings)]

provider_vectors = [(p['id'], emb.tolist(), {
    'type': 'scm',
    'name': p['name'],
    'id': p['id'],
    'data': json.dumps(p)  # Stringify nested dict
}) for p, emb in zip(providers, provider_embeddings)]

index.upsert(vectors=connector_vectors + provider_vectors)
print(f"Embeddings of full JSON stored in Pinecone index: {index_name}")