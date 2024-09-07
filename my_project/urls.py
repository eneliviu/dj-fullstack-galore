"""
URL configuration for my_project project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static

# The project-level urls.py file is the top level of our URLs.
# Include all app urls.py- files in the project urls.py file:
urlpatterns = [
    path('about/',
         include('about.urls'),
         name='about'),
    path("accounts/",
         include("allauth.urls")),
    path('admin/',
         admin.site.urls),
    path('leaflet_app/',
         include('leaflet_app.urls')),
    path('map_app/',
         include('map_app.urls'),
         name='map_url'),
    path('rag/',
         include('app_rag.urls'),
         name="index"),
    path('',
         include('landing.urls'),
         name="start"),
]

if settings.DEBUG:  # to serve media files during development
    urlpatterns += static(settings.MEDIA_URL,
                          document_root=settings.MEDIA_ROOT)
    urlpatterns += [
                    path('accounts/', include('django.contrib.auth.urls')),
                    ]


