from django.contrib import admin
from django.urls import path

from .views import home, result,image_upload_view

urlpatterns = [
    path('', home, name='home'),  
    path('result/', result, name='result'),
    path('upload/', image_upload_view),
 
]