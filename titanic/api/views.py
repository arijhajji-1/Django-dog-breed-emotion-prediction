from django.shortcuts import render
from django.http import JsonResponse
from .forms import ImageForm
from base.views import getPredictions
import json
import numpy as np
from django.http import HttpResponse
def predict_breed(request):
    if request.method == 'POST':  
        form = ImageForm(request.POST, request.FILES)
        if form.is_valid():
            img = request.FILES['image']
            print(img)
            labels = getPredictions(img)
           
            print("XXXXXXXXXXXXXXXXXXXXXXX",labels[0][0][1], " || ", labels[0][1][1], " || ", labels[0][2][1] )
            #resultPrediction = labels[0][0][1]+ " || "+ labels[0][1][1]+" || "+ labels[0][2][1]
            resultPrediction = labels[0][0][1]
            return HttpResponse(resultPrediction)
            
        else:
            return JsonResponse({'error': 'Invalid form data.'})
    else:
        return JsonResponse({'error': 'Invalid request method.'})

