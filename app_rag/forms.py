from django import forms
from .models import LoadImage


class DocumentForm(forms.Form):
    '''
    Form for uploading documents
    '''
    title = forms.CharField(max_length=100)
    file = forms.FileField()
    
    
class ImageLoadForm(forms.ModelForm):
    '''
    Form for loading images
    '''
    class Meta:
        model = LoadImage
        fields = ('name', 'image')
