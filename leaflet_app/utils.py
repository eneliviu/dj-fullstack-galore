from geopy import geocoders
from geopy.exc import GeocoderTimedOut


def get_coordinates(location, attempt=1, max_attempts=5):
    '''
    Geocodes an address with retry on timeout.
    GeoPy documentation: https://geopy.readthedocs.io/en/latest/#geopy.exc.GeocoderTimedOut
    
    Parameters:
    location (str): The location to geocode.
    attempts (int, optional): Current retry attempt. Default is 1.
    max_attempts (int, optional): Maximum retry attempts. Default is 5.
    
    Returns:
    Geocoded location data (Latitude and Longitude).
     
    Raises:
    GeocoderTimedOut: If the max number of attempts is exceeded.
    '''
    geocoder = geocoders.Nominatim(user_agent='leaflet_app')
    try:
        location = geocoder.geocode(location, exactly_one=True, language='en')
        lat = geocoder.geocode(location).latitude
        lon = geocoder.geocode(location).longitude
        return lat, lon
    except GeocoderTimedOut:
        if attempt <= max_attempts:
            return get_coordinates(location,
                                   attempt=attempt+1)
        raise
    