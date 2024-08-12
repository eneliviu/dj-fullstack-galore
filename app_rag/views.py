import os
from django.shortcuts import render, get_object_or_404, redirect
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from django.http import HttpResponseRedirect
from pgvector.django import L2Distance
import uuid

from .summary_generator import generate_story
from .models import LangchainPgEmbedding, LoadImage
from .embedding import get_embedding
from .forms import DocumentForm

from dotenv import load_dotenv
load_dotenv('/home/lien/NLP/dj-fullstack-galore/app_rag/.env',
             override=True)

# Create your views here.


def index(request):
    '''
    View for RAG-app page
    '''
    return render(request, "app_rag/index.html")


def rag_dashboard(request):
    '''
    View for RAG-app page
    '''
    if request.method == "POST":
        text = request.POST.get('user-query-input')
      
        # create embedding from the text
        embedding = get_embedding(text)
        
        # create new embedding in the table
        # embedding_model = LangchainPgEmbedding.objects.create(
        #     uuid=uuid.uuid4(),
        #     embedding=embedding,
        #     document=text,
        # )
        # print(embedding_model.pk)
        
        doc = LangchainPgEmbedding.objects.all().order_by(L2Distance('embedding',
                                                                      embedding)).first()
        # TODO: OpenAI summarization:
        
        context = {'text': text,
                   'most_similar': doc
                   }
        
        return render(request,
                      "app_rag/rag_dashboard.html",
                      context)
        
    elif request.method == 'GET':
        return render(request, "app_rag/rag_dashboard.html")
      

def generate_story_from_words(request):
    '''
    Take the user input to the LLM:
    '''

    if request.method == 'GET':

        # Extract the expected words from the request
        words = request.GET.get('words')

        # Call the generate_story function with the extracted words
        answer = generate_story(words)
    else:
        pass

    return render(request,
                  "app_rag/rag_dashboard.html",
                  {'answer': answer}
                  )

# ----------- Simple upload -------------------------------#


def simple_upload(request):
    '''
    Simple file upload
    '''
    if request.method == 'POST' and request.FILES['myfile']:
        myfile = request.FILES['myfile']
        fs = FileSystemStorage()
        filename = fs.save(myfile.name, myfile)
        uploaded_file_url = fs.url(filename)
        return render(request, 
                      'app_rag/simple_upload.html',
                      {'uploaded_file_url': uploaded_file_url}
                      )
    return render(request,
                  'app_rag/simple_upload.html')
   

# ----------- Form upload -------------------------------#
def model_form_upload(request):
    '''
    Upload files using Django Forms
    Does not save the uploaded file
    '''
    import os
    from .utils import (load_pdf,
                        chunk_data,
                        make_embeddings_chroma,
                        get_session_history) 
    
    if request.method == "POST":
        form = DocumentForm(request.POST,
                            request.FILES)
        chunk_size = int(os.getenv('CHUNK_SIZE'))
        chunk_overlap = int(os.getenv('CHUNK_OVERLAP'))
        embedding_model = os.getenv('EMBEDDING_MODEL_DEFAULT')
        chroma_path = os.getenv('CHROMA_PATH')
        
        if form.is_valid():  # handle_uploaded_file(request.FILES["file"])         
            # LLM_DEFAULT = 'gpt-4o-2024-08-06'  # "gpt-3.5-turbo" #
            # EMBEDDING_MODEL_DEFAULT = 'text-embedding-3-large'
            # CHUNK_SIZE = 1000
            # CHUNK_OVERLAP = 250
            # CHROMA_PATH = './chroma_db'
            
            print('Loading...')
            docs = load_pdf(request.FILES["file"])
            
            print('Start document splitting...')
            splits = chunk_data(docs,
                                chunk_size=chunk_size,
                                chunk_overlap=chunk_overlap)
            
            print('Create vector embeddings...')
            vectorstore = make_embeddings_chroma(splits,
                                                 model_name=embedding_model,
                                                 persist_directory=chroma_path)
            return HttpResponseRedirect("/rag/rag_dashboard")
    else:
        print('The form is not valid!')
        form = DocumentForm()

    context = {
        "form": form
        }
    return render(request,
                  "app_rag/model_form_upload.html",
                  context
                  )


# Handling uploaded images with a model
def model_form_upload_images(request):
    '''
    View for uploading images
    '''
    from .forms import ImageLoadForm
    
    if request.method == 'POST':
        form = ImageLoadForm(request.POST,
                             request.FILES)
        if form.is_valid():
            form.save()
        else:
            context = {'form': form}
            return render(request,
                          'app_rag/model_form_upload.html',
                          context)
    
    context = {'form': ImageLoadForm()}
    
    return render(request,
                  'app_rag/model_form_upload.html',
                  context)
    

def delete_image(request, pk):
    '''
    Delete image
    '''
    img = get_object_or_404(LoadImage, pk=pk)
    img.delete()
    return redirect('app_rag/model_form_upload/')


def rag_chromadb(request):
    '''
    View for RAG-app page using Chromadb
    '''
    
    if request.method == "POST":
        
        from langchain.chains import (create_history_aware_retriever,
                                      create_retrieval_chain)
        from langchain.chains.combine_documents import create_stuff_documents_chain
        from langchain_community.chat_message_histories import ChatMessageHistory
        from langchain_core.chat_history import BaseChatMessageHistory
        from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
        from langchain_core.runnables.history import RunnableWithMessageHistory
        from langchain_openai import ChatOpenAI
        
        from .utils import load_embeddings_chroma, get_session_history
        
        embedding_model = os.getenv('EMBEDDING_MODEL_DEFAULT')
        chroma_path = os.getenv('CHROMA_PATH')
        llm_default = os.getenv('LLM_DEFAULT')
         
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

        llm = ChatOpenAI(model=llm_default,
                         temperature=0)
        vectorstore = load_embeddings_chroma(persist_directory=chroma_path,
                                             embedding_model_default=embedding_model)
        retriever = vectorstore.as_retriever()
        history_aware_retriever = create_history_aware_retriever(
            llm,
            retriever,
            contextualize_q_prompt
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
        
        store = {}
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
        )
        
        human_msg = conversational_rag_chain['input']
        ai_msg = conversational_rag_chain['answer']
        
        
        context = {'text': human_msg,
                   'most_similar': ai_msg
                   }
        
        return render(request,
                      "app_rag/rag_dashboard.html",
                      context)
        
    elif request.method == 'GET':
        return render(request, "app_rag/rag_dashboard.html")
      
