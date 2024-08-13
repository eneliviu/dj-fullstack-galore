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
    if text:
        text = text.replace("\n", " ")
    else:
        text = 'hi boot'
    input_vector = OpenAIEmbeddings(model=EMBEDDING_MODEL).embed_query(text)
    return input_vector


def load_document(file):
    '''
    Load file(s)
    '''
    import os
    _, extension = os.path.splitext(file)
    
    if extension == '.pdf':
        from langchain.document_loaders import PyPDFLoader
        print(f'Loading {file}')
        loader = PyPDFLoader(file)
    elif extension == '.docx':
        from langchain.document_loaders import Docx2txtLoader
        print(f'Loading {file}')
        loader = Docx2txtLoader(file)
    else:
        raise ValueError("Document format is not supported")
        
    data_from_file = loader.load()
    
    return data_from_file



def chunk_data(data, chunk_size=1000, chunk_overlap=200):
    '''
    Make chunks
    '''
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,
                                                   chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(data)
    # use ```create_documents``` when data is not splitted in pages
    
    return chunks

# Run:
# data = load_document('/home/lien/NLP/dj-fullstack-galore/Salas2024_point_patterns_thinnings.pdf')
# chunks = chunk_data(data)

def print_embedding_cost(texts):
    '''
    Calculate the OpenAI embedding costs
    '''
    import tiktoken
    enc = tiktoken.encoding_for_model(EMBEDDING_MODEL)
    total_tokens = sum([len(enc.encode(page.page_content)) for page in texts])
    print(f'Total Tokens: {total_tokens}')
    print(f'Embedding Cost $: {0.0004 * total_tokens / 1000:.6f}')
    

# Run:
# print_embedding_cost(chunks)


def qa_chain(vector_store, q):
    '''
    Asking and getting questions
    '''
    from langchain.chains import create_retrieval_chain
    from langchain.chains.combine_documents import create_stuff_documents_chain
    # from langchain.chains.retrieval_qa.base import RetrievalQA
    from langchain_openai import ChatOpenAI
    
    retriever = vector_store.as_retriever(search_type='similarity',
                                          search_kwargs={'k': 5})
    chat = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=1)
    
    system_prompt = (
        "Use the given context to answer the question. "
        "If you don't know the answer, say you don't know. "
        "Use three sentence maximum and keep the answer concise. "
        "Context: {context}"
    )
    
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )
    
    question_answer_chain = create_stuff_documents_chain(chat,
                                                         prompt)
    chain = create_retrieval_chain(retriever,
                                   question_answer_chain)
    answer = chain.invoke({"input": q})    
    return answer


def make_embeddings_chromadb(chunks, persist_directory='./chroma_db'):
    '''
    Use chroma db as vector store
    '''
    # from langchain_chroma import Chroma
    from langchain_community.vectorstores import Chroma
    from langchain_openai import ChatOpenAI
    from langchain_openai import OpenAIEmbeddings
 
    embedding_function = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    # chroma vector store object
    vector_store = Chroma.from_documents(chunks,
                                         embedding_function,
                                         persist_directory=persist_directory)
    return vector_store


def load_embeddings_chromadb(persist_directory='./chroma_db'):
    '''
    Load the existing embeddings to a vector store object
    '''
    from langchain_chroma import Chroma
    from langchain_openai import OpenAIEmbeddings
    
    embedding_function = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    
    vector_store = Chroma(persist_directory=persist_directory,
                          embedding_function=embedding_function)
    
    return vector_store



