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
    if request.method == 'POST':
        form = SearchForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('/map_app')
    else:
        form = SearchForm()
    
    address = Search.objects.all().last()  
    location = geocoder.osm(address)
    lat = location.lat
    lng = location.lng
    print([lat, lng])
    country = location.country
    if lat is None or lng is None:
        address.delete()
        return HttpResponse('Your address input is invalid')
    
    m = folium.Map(tiles="cartodb positron",
                   zoom_start=3,
                   control_scale=True)
    
    MousePosition().add_to(m)
    Draw(export=False,
         draw_options=True,
         position='bottomleft').add_to(m)
    Fullscreen(position='topright').add_to(m)
     
    folium.Marker([lat, lng],
                  tooltip='Click for more',
                  popup=country).add_to(m)
    
    context = {'map': m._repr_html_(),  # HTML representation of the Map() object
               'form': form}
    return render(
        request,
        'map_app/map.html',
        context
    )

