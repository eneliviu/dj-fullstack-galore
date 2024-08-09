import os
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv, find_dotenv

EMBEDDING_MODEL = 'text-embedding-ada-002'


def get_embedding(text: str):
    '''
    Make model embeddings from the user text query
    '''
    text = text.replace("\n", " ")
    input_vector = OpenAIEmbeddings(model_name=EMBEDDING_MODEL).embed_query(text)
    
    return input_vector

