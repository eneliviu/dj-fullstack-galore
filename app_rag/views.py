
from django.shortcuts import render, get_object_or_404, reverse
from pgvector.django import L2Distance
import uuid
from .summary_generator import generate_story
from .models import LangchainPgEmbedding
from .embedding import get_embedding


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
