from keras.models import load_model
classifier= load_model('model_xception.h5')

import numpy as np


from keras.preprocessing import image
from IPython.display import Image, display
def predict_gender(img_name):
    img_width, img_height = 224, 224
    img = image.load_img(img_name, target_size = (img_width, img_height))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis = 0)
    a=classifier.predict(img)
    a= a.flatten()

    if a[0]<0.5:

        print("Female ", a[0])
        return "Female"
    else:
        print("Male ", a[0])
        return "Male"

#testing images
a1 = predict_gender('kisolju.png')
print(a1)

a1 = predict_gender('asian man.jpg')
print(a1)

def load_caffe_models():
    age_net1 = cv2.dnn.readNetFromCaffe('deploy_age.prototxt', 'age_net.caffemodel')
    return age_net1

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
age_list = ['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)', '(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']

import cv2



def video_detector(age_net):

    i=0
    cap = cv2.VideoCapture(0+cv2.CAP_DSHOW)
    #cap.set(cv2.CV_CAP_PROP_FPS, 60)
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
    while (True):
        ret, frame = cap.read()    #return value if frame is available for capture and frame
        if ret==True:
            grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(grey, 1.1, 4)

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 3)
                crop_img = frame[y:y + h, x:x + w].copy()

                cv2.imwrite('frame' + str(i) + '.jpg', crop_img)
                blob = cv2.dnn.blobFromImage(crop_img, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
                age_net.setInput(blob)
                age_preds = age_net.forward()
                age = age_list[age_preds[0].argmax()]
                print("Age: " + age)


                text = predict_gender('frame'+str(i)+'.jpg') + age
                frame = cv2.putText(frame, text, org=(x,y), fontFace= cv2.FONT_HERSHEY_SIMPLEX,fontScale= 1, color=(255,255,0), thickness=2)
                cv2.imshow('frame', frame)
                i += 1


            if cv2.waitKey(50) & 0xFF ==ord('q'):
                break
        else:
            break
    cap.release()


if __name__ == "__main__":
    age_net = load_caffe_models()
    video_detector(age_net)
