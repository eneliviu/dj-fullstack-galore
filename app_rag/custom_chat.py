
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

# Q&A ChatBot:

# Load docs in Langchain:

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


# %%

# Wikipedia loader


def load_from_wikipedia(query, lang='sv', load_max_docs=2):
    from langchain.document_loaders import WikipediaLoader
    loader = WikipediaLoader(query=query,
                             lang=lang,
                             load_max_docs=load_max_docs)
    data_from_wiki = loader.load()
    return data_from_wiki


data = load_from_wikipedia('GPT4')


# %%

data = load_document('/home/lien/NLP/dj-fullstack-galore/Salas2024_point_patterns_thinnings.pdf')
data = load_document('/home/lien/NLP/dj-fullstack-galore/Arbetsrapport dronare.docx')
print(data[1].page_content)
print(data[1].metadata)
print(len(data))
print(len(data[2].page_content))


# %%

# Make chunks


def chunk_data(data, chunk_size=256):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,
                                                   chunk_overlap=0)
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
    
# %%

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
            spec=PodSpec(environment='gpc-strater')
        )
        
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
        print('deleting all indexes....')
        for index in indexes:
            pc.describe_index(index)
    else:
        print(f'Deleting index {index_name}', end='')
        pc.delete_index(index_name)


     
