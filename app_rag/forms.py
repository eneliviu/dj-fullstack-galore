from django import forms


class DocumentForm(forms.Form):
    '''
    Form for uploading documents
    '''
    title = forms.CharField(max_length=100)
    file = forms.FileField()
