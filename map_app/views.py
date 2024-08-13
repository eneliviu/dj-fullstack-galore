from django.shortcuts import render
import folium
from folium.plugins import (Fullscreen, Draw, MousePosition)
# Create your views here.


def index(request):
    '''
    Create folium Map object
    '''
    m = folium.Map(tiles="cartodb positron",
                   zoom_start=3,
                   control_scale=True)
    MousePosition().add_to(m)
    
    draw = Draw(export=False, draw_options=True, position='bottomleft')
    draw.add_to(m)
    Fullscreen(position='topright').add_to(m)
     
    # Get HTML representation of the Map() object
    m = m._repr_html_()
    
    return render(
        request,
        'map_app/map.html',
        {'map': m}
    )

