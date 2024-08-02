from django.shortcuts import render
from django.http import HttpResponse


# Create your views here.
def about(request):
    '''
    View for About page
    '''
    # if request.method == "GET":
    #     return HttpResponse("This was a GET request from the About-page")
    # elif request.method == "POST":
    #     return HttpResponse("This was a POST request from the About-page")

    return render(request, "about/about.html")