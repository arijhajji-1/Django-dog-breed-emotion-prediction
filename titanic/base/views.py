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


size = 244

def read_and_prep_images(img_paths):
    imgs = [load_img(img_path, target_size=(224, 224)) for img_path in img_paths]
    img_array = np.array([img_to_array(img) for img in imgs])
    output = preprocess_input(img_array)
    return(output)



    
def home(request):
    return render(request, 'index.html')

def getPredictions():
    image_dir = 'C:\\Users\\Asma Ben Boubaker\\Desktop\\TitanicPredictionDjangoML\\titanic\\titanic\\train\\'
    
    img_paths = [join(image_dir, filename) for filename in 
                            ['1.jpg',
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
