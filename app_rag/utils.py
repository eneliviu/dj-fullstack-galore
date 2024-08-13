from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI

from dotenv import load_dotenv
load_dotenv('/home/lien/NLP/dj-fullstack-galore/app_rag/.env',
             override=True)


def load_pdf(file):
    from langchain.document_loaders import PyPDFLoader
    print(f'Loading {file}')
    loader = PyPDFLoader(file)
    data_from_file = loader.load()
    return data_from_file


def chunk_data(data, chunk_size, chunk_overlap):
    '''
    Split text in chunks
    '''
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,
                                                   chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(data)
    return chunks


def make_embeddings_chroma(chunks,
                           model_name,
                           persist_directory):
    '''
    Make embeddings and use Chromadb as vector store
    '''
    # from langchain_chroma import Chroma
    from langchain_community.vectorstores import Chroma
    from langchain_openai import OpenAIEmbeddings
 
    embedding_function = OpenAIEmbeddings(model=model_name)
    vector_store = Chroma.from_documents(chunks,
                                         embedding_function,
                                         collection_name="vectors",
                                         persist_directory=persist_directory)
    return vector_store


def load_embeddings_chroma(persist_directory, embedding_model_default):
    '''
    Load the existing embeddings to a vector store object
    '''
    from langchain_chroma import Chroma
    from langchain_openai import OpenAIEmbeddings
    
    embedding_function = OpenAIEmbeddings(
        model=embedding_model_default
        )
    vector_store = Chroma(persist_directory=persist_directory,
                          embedding_function=embedding_function)
    return vector_store

store = {}
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    '''
    Statefully manage chat history
    '''
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]