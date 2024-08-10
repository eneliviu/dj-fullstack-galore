from django.db import models
from django.contrib.auth.models import AbstractUser

# Create model 
class User(AbstractUser):
    cv = models.FieldFile(upload_to='cvs/',
                          null=True,
                          blank=True)
    upladed_at = models.DateTimeField(null=True,
                                      blank=True)
    
    
    # In settings.py:
    # AUTH_USER_MODEL = 'core.User'
    # MEDIA_ROOT = BASE_DIR / 'media'
    # MEDIA_URL = '/media/'
    
    # pip install djangorestframework
    
    
class Dog(models.Model):
    name = models.CharField(max_length=100)
    image = models.ImageField(upload_to='dogs/')
    
    def delete(self):
        self.image.delete()
        super().delete()
        
    

# in Views.py
from django.shortcuts import render, get_object_or_404, redirect
from .forms import DogForm
from .models import Dog

def upload_form(request):
    if request.method == 'POST':
        form = DogForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
    else:
        context = {'form': form}
        return render(request, 'core/index.html', context)
    
    context = {'form': DogForm()}
    return render(request,
                  'core/index.html',
                  context)
    
# pip install django-widget-tweaks? crispy-forms?


def list_dogs(request):
    '''
    Render the list of dogs
    '''
    dogs = Dog.objects.all()
    context = {'dogs': dogs}
    return render(request, 'core/list.html', context) # see ulrs.py

def delete_img(request, pk):
    dog = get_object_or_404(Dog, pk=pk)
    dog.delete()
    return redirect('list-view')
    # see django-cleanup to remove the files from storage
    
    

# go to urls.py: 
urlspatterns = [
    path('', views.upload_form, name='upload-form'),
    path('list/', views.list_dogs, name='list-view'),
    path('delete/<int_pk>', views.delete_img, name='delete-image'),
]
