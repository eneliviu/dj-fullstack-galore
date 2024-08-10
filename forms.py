from django import forms
from app_rag.models import Dog


class DogForm(forms.ModelForm):
    class Meta:
        model = Dog
        fields = ('name', 'image')
        