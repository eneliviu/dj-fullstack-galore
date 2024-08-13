from django.urls import path
from . import views as map_app_views

urlpatterns = [
    path('',
         map_app_views.index,
         name='map_app'),
]
