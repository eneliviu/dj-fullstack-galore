from django.shortcuts import render, get_object_or_404, reverse
from django.views import generic
from django.contrib import messages
from django.http import JsonResponse, HttpResponseRedirect
from .story_generator import generate_story


# Create your views here.
def index(request):
    '''
    View for RAG-app page
    '''
    return render(request, "app_rag/index.html")


# Create your views here.

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
                  "app_rag/index.html",
                  {'answer': answer}
                  )
