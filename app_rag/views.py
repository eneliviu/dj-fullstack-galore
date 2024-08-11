
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
    '''
  
    if request.method == "POST":
        form = DocumentForm(request.POST,
                            request.FILES)
        if form.is_valid():
            # handle_uploaded_file(request.FILES["file"])
            return HttpResponseRedirect("/rag/rag_dashboard")
    else:
        form = DocumentForm()
    return render(request,
                  "app_rag/model_form_upload.html",
                  {"form": form}
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