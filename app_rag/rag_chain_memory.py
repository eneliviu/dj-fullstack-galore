# %%
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI

from dotenv import load_dotenv, find_dotenv
load_dotenv('/home/lien/NLP/dj-fullstack-galore/app_rag/.env', 
            override=True)

# %%

LLM_DEFAULT = 'gpt-4o-2024-08-06' # "gpt-3.5-turbo" #
EMBEDDING_MODEL_DEFAULT = 'text-embedding-3-large'
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 250


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


def chunk_data(data,
               chunk_size=CHUNK_SIZE,
               chunk_overlap=CHUNK_OVERLAP):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,
                                                   chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(data)  # use create_documents when data is not splitted in pages
    
    return chunks


def make_embeddings_chroma(chunks,
                           model_name=EMBEDDING_MODEL_DEFAULT,
                           persist_directory='./chroma_db'):
    '''
    Use chroma db as vector store
    '''
    # from langchain_chroma import Chroma
    from langchain_community.vectorstores import Chroma
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
 
    embedding_function = OpenAIEmbeddings(model=model_name)
    # chroma vector store object
    vector_store = Chroma.from_documents(chunks,
                                         embedding_function,
                                         persist_directory=persist_directory)
    return vector_store


def load_embeddings_chroma(persist_directory='./chroma_db'):
    '''
    Load the existing embeddings to a vector store object
    '''
    from langchain_chroma import Chroma
    from langchain_openai import OpenAIEmbeddings
    
    embedding_function = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    
    vector_store = Chroma(persist_directory=persist_directory,
                          embedding_function=embedding_function)
    
    return vector_store


def format_human_msg(h_msgs: list[str]) -> list[str]:
    '''
    Parse HumanMessage
    '''
    q = [str(s).replace("'", '').replace('content=', '') for s in h_msgs]
    return q


def format_ai_answers(ai_msg: list[str]) -> list[str]:
    '''
    Parse AIMessage
    '''
    a = [str(s).replace("'", '').replace('content=', '') for s in ai_msg]
    return a


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# %%

llm = ChatOpenAI(model=LLM_DEFAULT, temperature=0)

# Transform loaders to Langchain data model
docs = load_document('/home/lien/NLP/dj-fullstack-galore/Salas2024_point_patterns_thinnings.pdf')

# Construct retriever ###
splits = chunk_data(docs,
                    chunk_size=CHUNK_SIZE,
                    chunk_overlap=CHUNK_OVERLAP)

vectorstore = make_embeddings_chroma(splits,
                                     model_name=EMBEDDING_MODEL_DEFAULT,
                                     persist_directory='./chroma_db')
retriever = vectorstore.as_retriever()

# %%

# CREATE RAG-CHAIN

# Contextualize question
# This chain prepends a rephrasing of the input query to our retriever,
# so that the retrieval incorporates the context of the conversation.

contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

# Answer question ###

qa_system_prompt = """You are an assistant for question-answering tasks. \
Use the following pieces of retrieved context to answer the question. \
If you don't know the answer, just say that you don't know. \
Use three sentences maximum and keep the answer concise.\

{context}"""
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever,
                                   question_answer_chain)


conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

# %%
# Statefully manage chat history ###

store = {}
out = []
out.append(conversational_rag_chain.invoke(
    {"input": "What is the document about?"},
    config={
        "configurable": {"session_id": "abc123"}
    },  # constructs a key "abc123" in `store`.
))


out.append(conversational_rag_chain.invoke(
    {"input": "How many authors the document has?"},
    config={"configurable": {"session_id": "abc123"}},
))

# %% PARSE CHAT HISTORY:

for elem in out:
    hst = elem['chat_history']
    if not hst:
        human_msg = elem['input']
        ai_msg = elem['answer']
    else:
        human_msg = hst[::2]
        ai_msg = hst[1::2]
        print(format_human_msg(human_msg))
        print(format_ai_answers(ai_msg))
  
# %% CLEAN-UP

vectorstore.delete_collection()

# %%
