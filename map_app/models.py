from django.db import models


# Create your models here.
class Search(models.Model):
    '''
    Adress geocoding class
    '''
    address = models.CharField(max_length=100, null=True)
    date = models.DateTimeField(auto_now_add=True)
    
    
    def __str__(self):
        return self.address
    