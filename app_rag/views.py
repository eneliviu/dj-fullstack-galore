
from django.shortcuts import render, get_object_or_404, reverse
from django.views import generic
from django.contrib import messages
from django.http import JsonResponse, HttpResponseRedirect
from .story_generator import generate_story
from .models import LangchainPgEmbedding
from .embedding import get_embedding
from pgvector.django import L2Distance


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
        text = request.POST.get('input_text')
      
        # create mebedding from the text
        embedding = get_embedding(text)
        obj = LangchainPgEmbedding.objects.all()
        document = obj.order_by(L2Distance('embedding', embedding)).first()
        
        # OpenAI summarization:
        
        context = {'text': text,
                   'most_similar': document
                   }
        return render(request, "app_rag/rag_dashboard.html", context)
      

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
