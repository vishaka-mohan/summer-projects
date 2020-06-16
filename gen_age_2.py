""" storing weights checkpoint"""
"""from tensorflow.python.keras.callbacks import ModelCheckpoint

checkpointer = ModelCheckpoint(filepath='gender_classification.weights.best.hdf5', verbose=1, save_best_only=True)"""

""" BUILDING CNN"""

# import keras libraries and packages
import keras
import tensorflow
from tensorflow.python.keras import Sequential

from tensorflow.python.keras.layers import Convolution2D
from tensorflow.python.keras.layers import MaxPooling2D
from tensorflow.python.keras.layers import Flatten
from tensorflow.python.keras.layers import Dense, Dropout
import numpy as np



# INITIALIZE CNN


classifier = Sequential()


# convolution
classifier.add(Convolution2D(96, 7, 7, input_shape=(192, 192, 3), activation='relu'))
classifier.add(Convolution2D(256, 5, 5, activation='relu'))
classifier.add(Convolution2D(384, 3, 3, activation='relu'))

# maxpooling
# classifier.add(MaxPooling2D(pool_size=(2,2)))

# Flatten
classifier.add(Flatten())

# full connection

classifier.add(Dense(512, activation='relu'))
classifier.add(Dropout(0.5))


classifier.add(Dense(512, activation='relu'))
classifier.add(Dropout(0.5))


classifier.add(Dense(1, activation='sigmoid'))
# compile
classifier.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])

# fitting cnn to image
from keras.preprocessing.image import ImageDataGenerator


train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)


test_datagen = ImageDataGenerator(rescale=1. / 255)


training_set = train_datagen.flow_from_directory(
    'train',
    target_size=(192, 192),
    batch_size=32,
    class_mode='binary')



test_set = test_datagen.flow_from_directory(
    'test',
    target_size=(192, 192),
    batch_size=32,
    class_mode='binary')



"""classifier.fit(
        training_set,
        steps_per_epoch=887,
        epochs=5,
        callbacks= [checkpointer],
        validation_data=test_set,
        validation_steps=273)"""

"""--------------------------------------------------------------"""

# load model
classifier.load_weights('gen_age_2_weights.hdf5')



# function to predict images
from keras.preprocessing import image
from IPython.display import Image, display



def predict_gender(img_name):
    
    img_width, img_height = 192, 192
    img = image.load_img(img_name, target_size=(img_width, img_height))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)

    
    a = classifier.predict(img)
    a = a.flatten()
    print(a[0])
    
    if a[0] < 0.5:
        print("Female ", a[0])
        return "Female"
    else:
        print("Male ", a[0])
        return "Male"


"""# testing two images
a1 = predict_gender('a.jpeg')
print(a1)

a1 = predict_gender('shiva_girly_photo.png')
print(a1)"""



def load_age_model():
    age_net1 = cv2.dnn.readNetFromCaffe('deploy_age.prototxt', 'age_net.caffemodel')
    return age_net1



MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
age_list = ['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)', '(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']


import cv2



def play_video(age_net):

    i=0
    cap = cv2.VideoCapture(0+cv2.CAP_DSHOW)
    #cap.set(cv2.CV_CAP_PROP_FPS, 60)

    #face detection using haar cascade
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
    while (True):
        ret, frame = cap.read()    #return value if frame is available for capture and frame
        if ret==True:
            grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(grey, 1.1, 4)

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 205, 165), 3)


                #cropping image


                crop_img = frame[y:y + h, x:x + w].copy()

                cv2.imwrite('frame' + str(i) + '.jpg', crop_img)
                blob = cv2.dnn.blobFromImage(crop_img, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
                age_net.setInput(blob)
                age_pred = age_net.forward()
                age = age_list[age_pred[0].argmax()]
                print("Age: " + age)


                text = predict_gender('frame'+str(i)+'.jpg') + age
                frame = cv2.putText(frame, text, org=(x,y), fontFace= cv2.FONT_HERSHEY_COMPLEX,fontScale= 1, color=(100,205,165), thickness=2)
                cv2.imshow('frame', frame)

                i += 1
                
            #q to quit
                


            if cv2.waitKey(50) & 0xFF ==ord('q'):
                break
        else:
            break
    cap.release()


if __name__ == "__main__":
    
    age_net = load_age_model()
    
    play_video(age_net)
