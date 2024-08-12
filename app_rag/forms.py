from django import forms
from .models import LoadImage


class DocumentForm(forms.Form):
    '''
    Form for uploading documents. No model required.
    '''
    title = forms.CharField(max_length=100, required=True)
    file = forms.FileField(required=True)
    

class ImageLoadForm(forms.ModelForm):
    '''
    Form for loading images. Requires a model in models.py
    '''
    class Meta:
        model = LoadImage
        fields = ('name', 'image')
