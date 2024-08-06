# %%
import os
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings

from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv,  find_dotenv


# %%

loader = TextLoader('state_of_the_union.txt', encoding='utf-8')
documents = loader.load()

# %%
print(documents)
print(len(documents))
# %%

# Split text in chunks

text_splitter = RecursiveCharacterTextSplitter(chunk_size=100,
                                               chunk_overlap=20,
                                               length_function=len)
chunks = text_splitter.split_documents(documents)

# %%

# Calculate the embedding cost:


def embedding_cost(content):
    '''
    Calculate the cost of creating embeddings using a specific model
    '''
    import tiktoken
    tokens = tiktoken.encoding_for_model('text-embedding-ada-002')
    total_tokens = sum([len(tokens.encode(page.page_content))
                        for page in content])
    print(f'Total tokens: {total_tokens}')
    print(f'Embedding cost in USD: {total_tokens / 1000 * 0.0004:.6f}')


embedding_cost(chunks)

# %%

# Create embeddings from the chunks
load_dotenv('/home/lien/NLP/dj-fullstack-galore/app_rag/.env')

embeddings = OpenAIEmbeddings()
vector = embeddings.embed_query('testing the embedding model')

# %%

document_vectors = embeddings.embed_documents([t.page_content for t in chunks[:5]])

# %%

# PGvector

from langchain_postgres import PGVector

CONNECTION_STRING = "postgresql+psycopg:///postgres"
COLLECTION_NAME = 'state_of_the_union'

db = PGVector.from_documents(embedding=embeddings,
                             documents=chunks,
                             collection_name=COLLECTION_NAME,
                             connection=CONNECTION_STRING,
                             use_jsonb=True,)
retriever  = db.as_retriever(
    search_type="mmr",
    search_kwargs={'k': 4,
                   'lambda_mult': 0.25,
                   'score_threshold': 0.5}
)
# %%

query = 'What did the president say about Russia'
similar = db.similarity_search_with_score(query)

for doc in similar:
    print(doc)

# %%
similar = retriever.invoke(query)
for doc in similar:
    print(doc)
    
# %%
db.drop_tables()

# %%

