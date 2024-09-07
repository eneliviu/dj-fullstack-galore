from geopy import geocoders
from django.shortcuts import render, get_object_or_404, reverse, redirect
from django.views import generic
from django.contrib import messages
from django.http import HttpResponseRedirect

from . models import Trip
from .forms import TripForm
 

# Create your views here.
def index(request):
       
    if request.method == 'GET':
        if Trip.objects.all().last() is None:
            form = TripForm()
            context = {'form': form}
        else:
            trips = list(Trip.objects.values())
            print(trips)
            
            form = TripForm()
            context = {'form': form,
                       'trips': trips}
            return render(request, 
                        'leaflet_app/leaflet_map.html',
                        context)
  
    if request.method == 'POST':
        # geocoder = geocoders.Nominatim(user_agent='my_app')
        # coords = [(geocoder.geocode(loc['location']).latitude,
        #            geocoder.geocode(loc['location']).longitude)
        #          for loc in list(Trip.objects.values())]
        # lat = [c[0] for c in coords]
        # lon = [c[1] for c in coords]
        
        form = TripForm(request.POST)
        if form.is_valid():
            trip = form.save(commit=False)
            trip.tourist = request.user
            form.save()
            messages.add_message(
                request,
                messages.SUCCESS,
                'Trip info succesfully added'
            )
            
        else:
            print(form.errors.as_data())
            messages.add_message(
                request,
                messages.ERROR,
                form.errors.as_data()
            )
            form = TripForm()
        context = {'form': form,
                   'trips': trip}
    
    return render(request, 
                  'leaflet_app/leaflet_map.html',
                  context)
