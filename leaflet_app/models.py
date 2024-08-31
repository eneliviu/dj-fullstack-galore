from django.db import models
from django.core.validators import MaxValueValidator as mxvv
from django.contrib.auth import get_user_model


User = get_user_model()

# Create your models here.
class Trip(models.Model):
    country = models.CharField(max_length=20)
    location = models.CharField(max_length=100)
    start_date = models.DateField(auto_now_add=False)
    end_date = models.DateField(auto_now_add=False)
    tourist = models.ForeignKey(User,
                                on_delete=models.CASCADE,
                                related_name='trip')
    lat = models.FloatField(blank=True, null=True)
    lon = models.FloatField(blank=True, null=True)
    
    def __str__(self):
        return self.location


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
        return f"{self.name} in {self.journey.location}"
