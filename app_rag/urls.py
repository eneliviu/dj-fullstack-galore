from . import views as app_rag_views
from django.urls import path


urlpatterns = [
    path('', app_rag_views.index, name='index'),
]
