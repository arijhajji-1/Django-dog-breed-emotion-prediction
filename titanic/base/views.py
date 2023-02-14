from django.shortcuts import render
import pickle
import numpy as np
from os.path import join
from keras.applications.resnet  import preprocess_input
#from keras.preprocessing.image import  img_to_array
from keras.utils import load_img, img_to_array
from keras.applications import ResNet50
from learntools.deep_learning.decode_predictions import decode_predictions
from IPython.display import Image, display
from django.shortcuts import render
from .forms import ImageForm
size = 244

def read_and_prep_images(img_paths):
    imgs = [load_img(img_path, target_size=(224, 224)) for img_path in img_paths]
    img_array = np.array([img_to_array(img) for img in imgs])
    output = preprocess_input(img_array)
    return(output)



    
def home(request):
    return render(request, 'index.html')

def getPredictions(img):
    image_dir = 'D:\\4twin\\s2\\pi\\Django-dog-breed-prediction\\titanic\\media\\images'
    
    img_paths = [join(image_dir, filename) for filename in 
                            [img,
                            ]]
    image_size = 224
    #read_and_prep_images(img_paths, 244, 244)
    my_model = ResNet50(weights='imagenet')
    test_data = read_and_prep_images(img_paths)
    preds = my_model.predict(test_data)
    
   
    most_likely_labels = decode_predictions(preds, top=3, class_list_path='C:\\Users\\Asma Ben Boubaker\\input\\ResNet-50\\imagenet_class_index.json')
    for i, img_path in enumerate(img_paths):
        display(Image(img_path))
        print(most_likely_labels[i])
    return most_likely_labels

def result(request):
    result = getPredictions()

    return render(request, 'result.html', {'result': result})
def image_upload_view(request):
    """Process images uploaded by users"""
    if request.method == 'POST':
        form = ImageForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            # Get the current instance object to display in the template
            img_obj = form.instance
            image_name = img_obj.image.name.split('/')[-1]
            print(image_name)
            result = getPredictions(image_name)
            print(result)
            return render(request, 'index.html', {'form': form, 'img_obj': img_obj,'result':result})
    else:
        form = ImageForm()
    return render(request, 'index.html', {'form': form})