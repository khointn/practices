import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt


#### Import model:
new_model = tf.keras.models.load_model(r"C:\Users\nguye\Downloads\cnn_model.h5")

path = "haarcascade_frontalface_default.xml"
font_scale = 1.5
font = cv2.FONT_HERSHEY_PLAIN

### Set rectangle background
rectangle_brg = (255,255,255)
img = np.zeros((500,500))
text = "some text in the box"
text_width, text_height = cv2.getTextSize(text, font, fontScale = font_scale, thickness = 1)[0]

### Set the text start position
text_offset_x = 10
text_offset_y = img.shape[0] -25

### make the coordinates of the box with a small padding of 2 pixel
box_coords = ((text_offset_x, text_offset_y),(text_offset_x+text_width+2, text_offset_y - text_height -2))
cv2.rectangle(img, box_coords[0], box_coords[1], rectangle_brg, cv2.FILLED)
cv2.putText(img, text, (text_offset_x, text_offset_y), font, fontScale = font_scale, color = (0,0,0), thickness = 1)

cap = cv2.VideoCapture(0)
### Check if webcam is open
if not cap.isOpened():
    cap = cv2.VideoCapture(1)
if not cap.isOpened():
    raise IOError("Cannot open webcam")

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + path)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.1, 4)

    for x,y,w,h in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)
        faces_1 = faceCascade.detectMultiScale(roi_gray)

        if len(faces_1) == 0:
            print("Face not detected!")
        else:
            for ex,ey,ew,eh in faces_1:
                face_roi = roi_color[ey:ey+eh, ex:ex+ew] ### crop face

    final_image = cv2.resize(face_roi, (224,224))
    final_image = np.expand_dims(final_image, axis = 0) ### add the 4th dimension
    #final_image = tf.keras.applications.imagenet_utils.preprocess_input(final_image,  mode='torch')

    #plt.imshow(final_image)
    #font = cv2.FONT_HERSHEY_SIMPLEX

    predictions = new_model.predict(final_image)

    status = ""

    if np.argmax(predictions) == 0:
        status = "Angry"
    elif np.argmax(predictions) == 1:
        status = "Disgust"
    elif np.argmax(predictions) == 2:
        status = "Fear"
    elif np.argmax(predictions) == 3:
        status = "Happy"
    elif np.argmax(predictions) == 4:
        status = "Sad"
    elif np.argmax(predictions) == 5:
        status = "Surprise"
    else:
        status = "Neutral"

    x1,y1, w1, h1 = 0, 0, 175, 75

    print(status, predictions)

    ###Draw background rectangle
    cv2.rectangle(frame, (x1, x1), (x1+w1, y1 + h1), (0,0,0), -1)
    ###Add text
    cv2.putText(frame, status, (x1 +int(w1/10), y1 + int(h1/2)), font, 0.7, (0,0,255), 2)
    cv2.putText(frame, status, (100, 150), font, 3, (0,0,255), 2, cv2.LINE_4)
    cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255))


    cv2.imshow('Face Emotion Recognition', frame)

    if cv2.waitKey(2) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
