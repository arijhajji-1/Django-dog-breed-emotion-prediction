from django.urls import path
from .views import predict_breed

urlpatterns = [
    path('predict/', predict_breed, name='predict'),
]


   