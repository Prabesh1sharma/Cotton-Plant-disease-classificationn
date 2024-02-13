from django.shortcuts import render
from django.http import HttpResponse
from django.conf import settings
import os
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model

# Load the model
model = load_model("model\cottondiseaseclassificationmodel.h5")

def pred_cot_dieas(cott_plant):
    test_image = load_img(cott_plant, target_size=(150, 150))
    print("@@ Got Image for prediction")

    test_image = img_to_array(test_image) / 255
    test_image = np.expand_dims(test_image, axis=0)

    result = model.predict(test_image).round(3)
    print('@@ Raw result = ', result)

    pred = np.argmax(result)

    if pred == 0:
        return "Healthy Cotton Plant", 'healthy_plant_leaf.html'
    elif pred == 1:
        return 'Diseased Cotton Plant', 'disease_plant.html'
    elif pred == 2:
        return 'Healthy Cotton Plant', 'healthy_plant.html'
    else:
        return "Healthy Cotton Plant", 'healthy_plant.html'

def home(request):
    return render(request, 'index.html')

def predict(request):
    if request.method == 'POST':
        try:
            file = request.FILES['image']
            filename = file.name
            print("@@ Input posted =", filename)

            # file_path = os.path.join(settings.MEDIA_ROOT, 'user_uploaded', filename)
            file_path = os.path.join('static/user uploaded', filename)
            
            if not os.path.exists(file_path):
                os.makedirs(os.path.dirname(file_path), exist_ok=True)

            with open(file_path, 'wb') as f:
                for chunk in file.chunks():
                    f.write(chunk)

            print("@@ Predicting class......")
            pred, output_page = pred_cot_dieas(cott_plant=file_path)

            # Pass 'user_image' key with the correct variable name to the template
            return render(request, output_page, {'pred_output': pred, 'user_image': file_path})
        except:
            return HttpResponse("<h1>OPPS!!!! You haven't the enter the image </h1><p><h2>Go back and Firstly enter the image </h2></p>")

    
    return HttpResponse("POST method required.")
