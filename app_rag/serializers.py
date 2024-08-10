from rest_framework import serializers
from .models import User

class CVUploadSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ['cv']
        
    
# in views.py
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView
from django.shortcuts import render
from .serializers import CVUploadSerializer
from .models import User

class UploadCVView(APIView):
    serializer_class = CVUploadSerializer
    
    def post(self, request):
        # assume that request.user is sent to the authenticated user
        request.user = User.objects.first()
        serializer = self.serializer_class(request.user,
                                           data=request.data)
        #request.POST
