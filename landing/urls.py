from . import views as landing_views
from django.urls import path


urlpatterns = [
    path('', landing_views.index, name='start'),
]