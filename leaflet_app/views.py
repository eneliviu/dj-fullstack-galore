
from geopy import geocoders
from django.shortcuts import render
from . models import Trip

# Create your views here.
def index(request):
    geocoder = geocoders.Nominatim(user_agent='my_app')
    coords = [(geocoder.geocode(loc['location']).latitude,
               geocoder.geocode(loc['location']).longitude) 
              for loc in list(Trip.objects.values('location'))]
    print(coords)

    #lat = coords.latitude
    #lon = coords.longitude
    #print([lat, lon])
    trips = list(Trip.objects.values("lat", "lon"))
    context = {'trips': trips}
    return render(request, 'leaflet_app/leaflet_map.html', context)
