
# %%
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import SystemMessage
from langchain.chains.llm import LLMChain 
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory, FileChatMessageHistory

# %%
from dotenv import load_dotenv,  find_dotenv
load_dotenv('/home/lien/NLP/dj-fullstack-galore/app_rag/.env', 
            override=True)

LLM = 'gpt-4o-2024-08-06'
EMBEDDING_MODEL = 'text-embedding-3-small'

# %%

llm = ChatOpenAI(model_name=LLM, temperature=1)

history = FileChatMessageHistory('chat_history.json')

memory = ConversationBufferMemory(
    memory_key='chat_history',
    chat_memory=history,
    return_messages=True
)


prompt = ChatPromptTemplate(
    input_variables=['content'],
    messages=[
        SystemMessage(content='You are a chatbot chatting with a human.\
            Respond only in Swedish'),
        MessagesPlaceholder(variable_name='chat_history'),  # store memory
        HumanMessagePromptTemplate.from_template('{content}')
    ]
)

chain = LLMChain(
    llm=llm,
    prompt=prompt,
    memory=memory,
    verbose=True,
)

# %%

while True:
    content = input('Your prompt: ')
    if content in ['quit', 'exit', 'bye']:
        print('Goodbye')
        break
    response = chain.run({'content': content})
    print(response)
    print('-' * 50)

# %%

# Transform loaders to Langchain data model


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


# Wikipedia loader

def load_from_wikipedia(query, lang='sv', load_max_docs=2):
    from langchain.document_loaders import WikipediaLoader
    loader = WikipediaLoader(query=query,
                             lang=lang,
                             load_max_docs=load_max_docs)
    data_from_wiki = loader.load()
    return data_from_wiki


# data = load_from_wikipedia('GPT4')


# %%

data = load_document('/home/lien/NLP/dj-fullstack-galore/Salas2024_point_patterns_thinnings.pdf')
# data = load_document('/home/lien/NLP/dj-fullstack-galore/Arbetsrapport dronare.docx')
print(data[1].page_content)
print(data[1].metadata)
print(len(data))
print(len(data[2].page_content))


# %%

# Make chunks


def chunk_data(data, chunk_size=1000, chunk_overlap=500):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,
                                                   chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(data)  # use create_documents when data is not splitted in pages
    
    return chunks

# %%
data = load_document('/home/lien/NLP/dj-fullstack-galore/Salas2024_point_patterns_thinnings.pdf')
chunks = chunk_data(data)
print(len(chunks))

# %%

def print_embedding_cost(texts):
    '''
    Calculate the OpenAI embedding costs
    '''
    import tiktoken
    enc = tiktoken.encoding_for_model(EMBEDDING_MODEL)
    total_tokens = sum([len(enc.encode(page.page_content)) for page in texts])
    print(f'Total Tokens: {total_tokens}')
    print(f'Embedding Cost $: {0.0004 * total_tokens / 1000:.6f}')
    
print_embedding_cost(chunks)

# %%  PINECONE
# Upload the chunks to database:

def insert_or_fetch_embeddings(index_name, chunks):
    import pinecone
    from langchain_community.vectorstores import Pinecone
    from langchain_openai import OpenAIEmbeddings
    from pinecone import PodSpec
    
    pc = pinecone.Pinecone()
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL,
                                  dimensions=1536)
    if index_name in pc.list_indexes().names():
        # Load embeddings if incex alreasy exists
        print(f'Index {index_name} already exists. Loading embeddings...')
        vector_store = Pinecone.from_existing_index(index_name,
                                                    embeddings)
        print('OK')
    else:
        print(f'Creating index {index_name} and embeddings ...', end='')
        pc.create_index(
            name=index_name,
            dimension=1536,
            metric='cosine',
            spec=PodSpec(
                environment='gcp-starter'
                )
        )
        # create vector store:
        vector_store = Pinecone.from_documents(chunks,
                                               embeddings, 
                                               index_name=index_name
                                               )
        print('OK')
        
    return vector_store

# Delete pinecone index 
def delete_pinecone_index(index_name='all'):
    '''
    Delete Free Tier Pincecone indexes (max one index): 
    '''
    import pinecone
    pc = pinecone.Pinecone()
    if index_name == 'all':
        indexes = pc.list_indexes().names()
        print('Deleting all indexes....')
        for index in indexes:
            print(index)
            pc.describe_index(index)
    else:
        print(f'Deleting index {index_name}', end='')
        print(pc.list_indexes().names())
        pc.delete_index(index_name)



# %% PINECONE RUN

index_name = 'askadocument'
vector_store = insert_or_fetch_embeddings(index_name, chunks)

q = 'What is the document about?'
answer = ask_and_get_answer(vector_store, q)
print(answer['answer'])
delete_pinecone_index(index_name)


# %%

# Ask and get questions

def ask_and_get_answer(vector_store, q):
    '''
    Asking and getting questions
    '''
    from langchain.chains import create_retrieval_chain
    from langchain.chains.combine_documents import create_stuff_documents_chain
    # from langchain.chains.retrieval_qa.base import RetrievalQA
    from langchain_openai import ChatOpenAI
    
    retriever = vector_store.as_retriever(search_type='similarity',
                                          search_kwargs={'k': 5})
    chat = ChatOpenAI(model_name=LLM, temperature=0)
    
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


# %%

import time
i = 1
print('Write Quite or Exit to quit.')
while True:
    q = input(f'Question #{i}: ')
    i += 1
    if q.lower() in ['quit', 'exit']:
        print('QUiting...bye bye!')
        time.sleep(2)
        
    answer = ask_and_get_answer(vector_store, q)
    print(f'\nAnswer: {answer['answer']}')
    print(f'\n{"-" * 50} \n')
    

# %%

data = load_from_wikipedia('ChatGPT', 'ro')
chunks = chunk_data(data)
index_name = 'chat_gpt'
vector_store = insert_or_fetch_embeddings(index_name)

q = 'Ce este ChatGPT'
answer = ask_and_get_answer(vector_store, q)
print(answer)


# %% 

# CHROMA DB

def create_embeddings_chroma(chunks, persist_directory='./chroma_db'):
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

   

# %%

data = load_document('/home/lien/NLP/dj-fullstack-galore/Salas2024_point_patterns_thinnings.pdf')
chunks = chunk_data(data, chunk_size=500, chunk_overlap=0)
print(len(chunks))

vector_store = create_embeddings_chroma(chunks)

# %%

q = 'What is the document about?'
answer = ask_and_get_answer(vector_store, q)
print(answer['answer'])

q = 'How many authors the document has?'
answer = ask_and_get_answer(vector_store, q)
print(answer['answer'])


# %%

db = load_embeddings_chroma()
answer = ask_and_get_answer(vector_store, q)
print(answer['answer'])
# cleanup
db.reset_collection()
db.delete_collection()

# %%

# Save chat histry and add memory
# Create custom prompt

from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate


# Structured Output: Pydantic clas
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import Optional
class Example(BaseModel):
    class Step(BaseModel):    
        question: Optional[str] = Field(default='What is the document about?',
                                        description="User question")
        answer: Optional[str] = Field(description="System answer")

    steps: list[Step]
    final_resolution: str = Field(
        description='The last chat answer'
    ) 

# Structured Output: TypedDict 
from typing_extensions import Annotated, TypedDict
class Example(TypedDict):
    """QA with memory."""
    question: Annotated[list[str], ..., "The user question"]
    answer: Annotated[list[str], ..., "The assistant answer"]

# 'gpt-4-turbo-preview'
llm = ChatOpenAI(model=LLM,
                 temperature=0)
structured_llm = llm#.with_structured_output(Example)

# vector_store = create_embeddings_chroma(chunks)
retriever = vector_store.as_retriever(search_type='similarity',
                                      search_kwargs={'k': 5})
memory = ConversationBufferMemory(memory_key='chat_history',
                                  return_messages=True)

system_template = r'''
                        Use the given context to answer the question. 
                        If you don't know the answer, say you don't know. 
                        Use three sentence maximum and keep the answer concise.
                        -----------------------------------
                        Context: ```{context}```
                   ''' 

user_template = r'''
                    Questions: ```{question}```
                    Chat History: ```{chat_history}```
                '''

messages = [
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template(user_template)
]

qa_prompt = ChatPromptTemplate.from_messages(messages)

# conversational retriever chain:
crc = ConversationalRetrievalChain.from_llm(
    llm=structured_llm,
    retriever=retriever,
    memory=memory,
    chain_type='stuff',
    combine_docs_chain_kwargs={'prompt': qa_prompt},
    verbose=True,
)


def ask_question(q, chain):
    '''
    Takes a question and returns am answer
    '''
    results = chain.invoke({'question': q})
    #results = chain.invoke(q)
    return results

q = 'How many authors the document has?'
results = ask_question(q, crc)  
print(results['answer'])


db = load_embeddings_chroma()
results = ask_question('What is the article about?',
                       crc)  
print(results['answer'])


memory.clear()
vector_store.delete_collection()

db = load_embeddings_chroma()
db.reset_collection()
db.delete_collection()


# %%



# %%

# Contextualizing the question
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

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

# Chain with chat history
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

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

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
# %%

from langchain_core.messages import HumanMessage

chat_history = []

question = "How many authors the document has?"
ai_msg_1 = rag_chain.invoke({"input": question, "chat_history": chat_history})
chat_history.extend([HumanMessage(content=question), ai_msg_1["answer"]])

second_question = "What are common ways of doing it?"
ai_msg_2 = rag_chain.invoke({"input": second_question, "chat_history": chat_history})
print(ai_msg_2["answer"])

# %%

from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from dotenv import load_dotenv,  find_dotenv
load_dotenv('/home/lien/NLP/dj-fullstack-galore/app_rag/.env', 
            override=True)

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)


### Construct retriever ###

docs = load_document('/home/lien/NLP/dj-fullstack-galore/Salas2024_point_patterns_thinnings.pdf')

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
retriever = vectorstore.as_retriever()


### Contextualize question ###
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


### Answer question ###
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

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)


### Statefully manage chat history ###
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

conversational_rag_chain.invoke(
    {"input": "What is the document about?"},
    config={
        "configurable": {"session_id": "abc123"}
    },  # constructs a key "abc123" in `store`.
)["answer"]

conversational_rag_chain.invoke(
    {"input": "How many authors the document has?"},
    config={"configurable": {"session_id": "abc123"}},
)["answer"]

