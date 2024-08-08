
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
    Load pdf file
    '''
    from langchain.document_loaders import PyPDFLoader
    print(f'Loading {file}')
    loader = PyPDFLoader(file)
    data = loader.load()
    return data


data = load_document('/home/lien/NLP/dj-fullstack-galore/Salas2024_point_patterns_thinnings.pdf')
print(data[1].page_content)
print(data[1].metadata)
print(len(data))
print(len(data[2].page_content))


# %%


