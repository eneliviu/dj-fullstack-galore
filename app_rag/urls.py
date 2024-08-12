from django.urls import path
from . import views as app_rag_views

urlpatterns = [
    path('',
         app_rag_views.index,
         name='index'),
    path('rag_dashboard/',
         app_rag_views.rag_dashboard,
         name='rag_dashboard'),
    path('simple_upload/',
         app_rag_views.simple_upload,
         name='simple_upload'),
    path('model_form_upload/',
         app_rag_views.model_form_upload,
         name='model_form_upload'),
]

