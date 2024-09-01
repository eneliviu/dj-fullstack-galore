from geopy import geocoders
from django.shortcuts import render, get_object_or_404, reverse
from django.views import generic
from django.contrib import messages
from django.http import HttpResponseRedirect

from . models import Trip
from .forms import TripForm
 

# Create your views here.
def index(request):
    
    queryset = Trip.objects.values()
    if not queryset:
        trips = get_object_or_404(queryset)
        print('Error')
    
    trips = list(trips)
        
    if request.method == 'POST':
        geocoder = geocoders.Nominatim(user_agent='my_app')
        coords = [(geocoder.geocode(loc['location']).latitude,
                geocoder.geocode(loc['location']).longitude)
                for loc in list(Trip.objects.values())]
        lat = [c[0] for c in coords]
        lon = [c[1] for c in coords]
        

        
        form = TripForm(request.POST)
        print(form.is_valid())
        if form.is_valid():
            form.save()
            messages.add_message(
                request,
                messages.SUCCESS,
                'Trip info succesfully added'
            )
        else:
            print('Error')
            form = TripForm()

    context = {'trips': trips,
               'form': TripForm}
    return render(request, 'leaflet_app/leaflet_map.html', context)
