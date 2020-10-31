import argparse
from PIL import Image 
import numpy as np 
import json
import tensorflow as tf
import tensorflow_hub as hub


parser = argparse.ArgumentParser()

parser.add_argument('--image_path')
parser.add_argument('--model' )
parser.add_argument('--top_k',default=5,type = int)
parser.add_argument('--class_label')
command = parser.parse_args()


image = command.image_path
top_k = command.top_k
my_model = command.model
class_label = command.class_label


with open('label_map.json', 'r') as f:
 class_names = json.load(f)
    
def process_image(image):
    img = np.squeeze(image)
    img =tf.image.resize(img,(224,224))
    img /=255
    return img.numpy()

def predict(image_path, model, top_k):
    img = Image.open(image_path)
    img = np.asarray(img)
    img = process_image(img)
    img = np.expand_dims(img, 0)
    predict = model.predict(img)
    probs,classes = tf.nn.top_k(predict, k=top_k)
    probs=probs.numpy()
    classes=classes.numpy()
    
    return probs,classes


if __name__ == '__main__':
    Flowers_model= tf.keras.models.load_model(my_model,custom_objects={'KerasLayer':hub.KerasLayer})


    probs,classes=predict(image, Flowers_model, top_k)
    #print name of flower
    cl_names =[]
    for i in classes[0]:
        classes_names=class_names[str(i+1)]
        cl_names.append(classes_names)
        
    #print result (class name, probablitiy of classes , class labels)    
    print("Name class is: ",cl_names[0])
    print("The probability of class: ", probs)
    print("The classes is: ", classes)