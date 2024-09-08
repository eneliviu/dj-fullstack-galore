
from django.db import models
from django.core.validators import MaxValueValidator as mxvv
from django.core.validators import ValidationError
from django.forms.forms import NON_FIELD_ERRORS
from django.contrib.auth.models import User
from .utils import get_coordinates

STATUS = ((0, "Completed"), (1, "Planned"), (2, 'Ongoing'))


# Create your models here.
class Trip(models.Model):
    '''
    Trip model
    '''
    country = models.CharField(max_length=20)
    location = models.CharField(max_length=100)
    start_date = models.DateField(auto_now_add=False)
    end_date = models.DateField(auto_now_add=False)
    tourist = models.ForeignKey(User,
                                on_delete=models.CASCADE,
                                related_name='trip')
    
    created_on = models.DateTimeField(auto_now_add=True)    
    status = models.IntegerField(choices=STATUS, default=0)
    
    lat = models.FloatField(blank=True, null=True)
    lon = models.FloatField(blank=True, null=True)
           
    def save(self, *args, **kwargs):
        '''
        Override the save() method to set the Lat and Lon values 
        before saving.
        '''
        try:
            coords = get_coordinates(self.location)
            self.lat = coords[0]
            self.lon = coords[1]
        except Exception as e:
            print(f"Operation failed: {e}")
    
        super(Trip, self).save(*args, **kwargs)
    
    # def validate_unique(self, *args, **kwargs):
    #     '''
    #     Override the validate_unique() model method to avoid
    #     overlapping trip dates.
        
    #     References:
    #     - https://shorturl.at/o5zXY, 
    #     - https://wiki.c2.com/?TestIfDateRangesOverlap
        
    #     '''
    #     super().validate_unique(*args, **kwargs)

    #     qs = self.__class__._default_manager.filter(
    #         start_date__lt = self.end_date,
    #         end_date__gt = self.start_date
    #     )

    #     if not self._state.adding and self.pk is not None:
    #         qs = qs.exclude(pk=self.pk)

    #     if qs.exists():
    #         raise ValidationError({
    #             NON_FIELD_ERRORS: ['overlapping date range',],
    #         })
    
    class Meta:
        ordering = ['start_date', 'country']
        
    def __str__(self):
        return f'{self.location}, {self.country}, {self.tourist}'
    

class Post(models.Model):
    TRIP_CATEG = (
        ('citybreaks', 'City breaks'),
        ('corporate', 'Corporate'),
        ('cultural', 'Cultural'),
        ('general', 'General'),
        ('holidays', 'Holidays'),
    )
    journey = models.ForeignKey(Trip,
                                on_delete=models.CASCADE,
                                related_name='post')
    name = models.CharField(max_length=100)
    description = models.TextField(max_length=500)
    category = models.CharField(max_length=100,
                                choices=TRIP_CATEG)
    image = models.ImageField(upload_to='post',
                              blank=True,
                              null=True)
    ratings = models.PositiveSmallIntegerField(default=1,
                                               validators=[mxvv(5)])

    def __str__(self):
        return f"{self.name} in {self.category}"
