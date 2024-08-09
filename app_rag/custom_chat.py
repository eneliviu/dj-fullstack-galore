
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
# %%

llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=1)

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


def chunk_data(data, chunk_size=1000, chunk_overlap=200):
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

encoding='utf-8'
def print_embedding_cost(texts):
    '''
    Calculate the OpenAI embedding costs
    '''
    import tiktoken
    enc = tiktoken.encoding_for_model('text-embedding-ada-002')
    total_tokens = sum([len(enc.encode(page.page_content)) for page in texts])
    print(f'Total Tokens: {total_tokens}')
    print(f'Embedding Cost $: {0.0004 * total_tokens / 1000:.6f}')
    
print_embedding_cost(chunks)

# %%

# Upload the chunks to database:

def insert_or_fetch_embeddings(index_name, chunks):
    import pinecone
    from langchain_community.vectorstores import Pinecone
    from langchain_openai import OpenAIEmbeddings
    from pinecone import PodSpec
    
    pc = pinecone.Pinecone()
    embeddings = OpenAIEmbeddings(model='text-embedding-3-small',
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



# %%

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
chinks = chunk_data(data)
index_name = 'chat_gpt'
vector_store = insert_or_fetch_embeddings(index_name)

q = 'Ce este ChatGPT'
answer = ask_and_get_answer(vector_store, q)
print(answer)


# %% 

# USE ChromaDB

def create_embeddings_chroma(chunks, persist_directory='./chroma_db'):
    '''
    Use chroma db as vector store
    '''
    from langchain_chroma import Chroma
    from langchain_openai import OpenAIEmbeddings
    
    embedding_function = OpenAIEmbeddings(model='text-embedding-ada-002')
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
    
    embedding_function = OpenAIEmbeddings(model='text-embedding-ada-002',
                                  dimensions=1536)
    
    vector_store = Chroma(persist_directory=persist_directory,
                          embedding_function=embedding_function)
    
    return vector_store

   

# %%

data = load_document('/home/lien/NLP/dj-fullstack-galore/Salas2024_point_patterns_thinnings.pdf')
chunks = chunk_data(data, chunk_size=1000, chunk_overlap=200)
print(len(chunks))

vector_store = create_embeddings_chroma(chunks)

# %%

q = 'What is the document about?'
answer = ask_and_get_answer(vector_store, q)
print(answer['answer'])
# %%

db = load_embeddings_chroma()
answer = ask_and_get_answer(vector_store, q)
print(answer['answer'])
# cleanup
vector_store.delete_collection()

# %%

# Save chat histry and add memory

from langchain_openai import ChatOpenAI
from langchain.chains import ConversationBufferMemory
from langchain.memory import ConversationBufferMemory

