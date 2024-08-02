from django.shortcuts import render
from django.http import HttpResponse


# Create your views here.
def index(request):
    '''
    View for RAG-app page
    '''

    # if request.method == "GET":
    #     return HttpResponse("This was a GET request for the RAG-app page")
    # elif request.method == "POST":
    #     return HttpResponse("This was a POST request for the RAG-app page")
    
    return render(request, "app_rag/index.html")