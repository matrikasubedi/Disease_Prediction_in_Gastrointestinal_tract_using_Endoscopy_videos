from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np 

model = load_model('/Users/matrikasubedi/Documents/Deep_Learning/Demo/kvasir.h5')

img_rows,img_cols = 224,224


class_labels = ['barretts', 'bbps-0-1', 'bbps-2-3', 'cecum', 'dyed-lifted-polyps', 'dyed-resection-margins', 'esophagitis-a', 'esophagitis-b-d', 'impacted-stool', 'polyps', 'pylorus', 'retroflex-rectum', 'retroflex-stomach', 'ulcerative-colitis-grade-0-1', 'z-line'] 
def check(path):
    
    
    # prediction
    img = image.load_img(path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x.astype('float32')/255
    pred = np.argmax(model.predict(x))
   
    print("It's a {}.".format(class_labels[pred])) 
  
check('/Users/matrikasubedi/Documents/Deep_Learning/Demo/static/images/Ulcerative Colitis.jpg')