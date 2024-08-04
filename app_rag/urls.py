from . import views as app_rag_views
from django.urls import path, include


urlpatterns = [
    path('', app_rag_views.index, name='index'),
    path('rag_dashboard/', app_rag_views.rag_dashboard,
         name='rag_dashboard'),
]

