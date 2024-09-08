from django.shortcuts import render, get_object_or_404, reverse, redirect
from django.views import generic
from django.contrib import messages
from django.http import HttpResponseRedirect

from . models import Trip
from .forms import TripForm
 

# Create your views here.

def index(request):
    print(request)
    
    if request.method == 'GET':
        # Handle no trip data:
        if Trip.objects.all().last() is None:
            form = TripForm()
            context = {'form': form}
        else:
            # Get all the trips:
            
            # List the trip values to serialize in Leaflet
            trips = list(Trip.objects.values())
            form = TripForm()
            context = {'form': form,
                       'trips': trips}
        return render(request, 
                      'leaflet_app/leaflet_map.html',
                      context)
  
    if request.method == 'POST':
        form = TripForm(request.POST)
        if form.is_valid():
            trip = form.save(commit=False)
            trip.tourist = request.user
            print(trip)
            form.save()
            messages.add_message(
                request,
                messages.SUCCESS,
                'Trip info succesfully added'
            )
            trip = list(Trip.objects.values())
            redirect('leaflet_map.html')
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
