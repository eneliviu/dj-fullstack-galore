from django import forms
from django.core.exceptions import ValidationError
from django.forms.forms import NON_FIELD_ERRORS
from .models import Trip


# create a ModelForm
class TripForm(forms.ModelForm):

    class Meta:
        model = Trip
        exclude = ['tourist', 'lat', 'lon', 'created_on', 'status']
        widgets = {
            'start_date': forms.widgets.DateInput(attrs={'type': 'date'}),
            'end_date': forms.widgets.DateInput(attrs={'type': 'date'})
        }
    
    # def clean(self):
    #     cleaned_data = super(TripForm, self).clean()
    #     start_date = cleaned_data.get('start_date')
    #     end_date = cleaned_data.get('end_date')
        
    #     if start_date and end_date:
    #         if end_date < start_date:
    #             self.add_error('end_date', 
    #                            'End date cannot be earlier than start date.')
         
    #     return cleaned_data
            
    # def clean(self, *args, **kwargs):
    #     '''
    #     Add custom validation to ensure that ```end_date``` is not
    #     prior to ```start_date```
    #     '''
    #     cleaned_data = super(TripForm, self).clean()
    #     start_date = cleaned_data.get('start_date')
    #     end_date = cleaned_data.get('end_date')
        
    #     if start_date and end_date: 
    #         if end_date < start_date:
    #             raise ValidationError({
    #                     NON_FIELD_ERRORS: ['overlapping date range',],
    #                 })
        
        # super().clean(*args, **kwargs)
