from django.shortcuts import render, redirect
import geocoder
import folium
from django.http import HttpResponse
from folium.plugins import (Fullscreen, Draw, MousePosition)
from .models import Search
from .forms import SearchForm

# Create your views here.


def index(request):
    '''
    Create folium Map object
    '''
    m = folium.Map(tiles="cartodb positron",
                   zoom_start=9,
                   control_scale=True)
    MousePosition().add_to(m)
    Draw(export=False,
         draw_options=True,
         position='bottomleft').add_to(m)
    Fullscreen(position='topright').add_to(m)
    
    if Search.objects.all().last() is None:
        if request.method == 'POST':
            folium.Marker([0, 0],
                        tooltip='Click for more',
                        popup='Default View').add_to(m)
            context = {'map': m._repr_html_(),
                       'form': SearchForm()}

    if request.method == 'POST':
        address_form = SearchForm(request.POST)
        if address_form.is_valid():             
            address_form.save()
            
        address = Search.objects.all().last() 
        location = geocoder.osm(address)
        lat = location.lat
        lng = location.lng
        print([lat, lng])
        country = location.country   
        if lat is None or lng is None:
            address.delete()
            context = {'map': m._repr_html_(),
                       'form': address_form} 
        else: 
            folium.Marker([lat, lng],
                          tooltip='Click for more',
                          popup=country).add_to(m)
            context = {'map': m._repr_html_(),
                       'form': address_form}
        
        return render(request,
                      'map_app/map.html',
                      context)
        
    else:
        return render(
            request,
            'map_app/map.html',
            {'form': SearchForm()}
        )
