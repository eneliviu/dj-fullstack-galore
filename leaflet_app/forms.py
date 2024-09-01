from django import forms
from .models import Trip
from django.core.exceptions import ValidationError

 
# create a ModelForm
class TripForm(forms.ModelForm):

    class Meta:
        model = Trip
        exclude = ['tourist', 'lat', 'lon']
        widgets = {
            'start_date': forms.widgets.DateInput(attrs={'type': 'date'}),
            'end_date': forms.widgets.DateInput(attrs={'type': 'date'})
        }
        
    def is_valid(self):
        valid = super(TripForm, self).is_valid()
        cleaned_data = super(TripForm, self).clean()
        if valid:
            if cleaned_data.get('start_date') < cleaned_data.get('end_date'):
                self.add_error('end_date',
                               'End date must be >= the start date')
                valid = False
        return valid
            
