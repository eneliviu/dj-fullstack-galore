from django.db import models


# Create your models here.
class Search(models.Model):
    '''
    Adress geocoding class
    '''
    address = models.CharField(max_length=100,
                               null=True,
                               unique=True,
                               help_text='Enter location...')
    date = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['address', 'date']
    
    verbose_name = 'Searches'
    
    # def get_absolute_url(self):
    #     """Returns the URL to access a particular instance of the model."""
    #     return reverse('model-detail-view', args=[str(self.id)])

    def __str__(self):
        return self.address
    