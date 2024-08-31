from django.urls import path
from . import views as leaflet_app_views

urlpatterns = [
    path('',
         leaflet_app_views.index,
         name='leaflet_app'),
]

