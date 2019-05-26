import cv2
import os
from time import sleep
import numpy as np
import argparse
from wide_resnet import WideResNet

depth = 16
width = 8
face_size = 64
model = WideResNet(face_size, depth=depth, k=width)()
model.load_weights("weights.18-4.06.hdf5")

def draw_label(image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX,
                font_scale=1, thickness=2):
    size = cv2.getTextSize(label, font, font_scale, thickness)[0]
    x, y = point
    cv2.rectangle(image, (x, y - size[1]), (x + size[0], y), (255, 0, 0), cv2.FILLED)
    cv2.putText(image, label, point, font, font_scale, (255, 255, 255), thickness)

def crop_face(imgarray, section, margin=40, size=64):
    img_h, img_w, _ = imgarray.shape
    if section is None:
        section = [0, 0, img_w, img_h]
    (x, y, w, h) = section
    margin = int(min(w,h) * margin / 100)
    x_a = x - margin
    y_a = y - margin
    x_b = x + w + margin
    y_b = y + h + margin
    if x_a < 0:
        x_b = min(x_b - x_a, img_w-1)
        x_a = 0
    if y_a < 0:
        y_b = min(y_b - y_a, img_h-1)
        y_a = 0
    if x_b > img_w:
        x_a = max(x_a - (x_b - img_w), 0)
        x_b = img_w
    if y_b > img_h:
        y_a = max(y_a - (y_b - img_h), 0)
        y_b = img_h
    cropped = imgarray[y_a: y_b, x_a: x_b]
    resized_img = cv2.resize(cropped, (size, size), interpolation=cv2.INTER_AREA)
    resized_img = np.array(resized_img)
    return resized_img, (x_a, y_a, x_b - x_a, y_b - y_a)


face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
video_capture = cv2.VideoCapture('../video.mp4')

while True:
    if bool(video_capture.isOpened())==False:
        sleep(5)

    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=10,
        minSize=(face_size, face_size)
    )
    face_imgs = np.empty((len(faces), face_size, face_size, 3))
    for i, face in enumerate(faces):
        face_img, cropped = crop_face(frame, face, margin=40, size=face_size)
        (x, y, w, h) = cropped
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 200, 0), 2)
        face_imgs[i,:,:,:] = face_img
    if len(face_imgs) > 0:
        results = model.predict(face_imgs)
        predicted_genders = results[0]
        ages = np.arange(0, 101).reshape(101, 1)
        predicted_ages = results[1].dot(ages).flatten()
    for i, face in enumerate(faces):
        label = "{}, {}".format(int(predicted_ages[i]),
                                "F" if predicted_genders[i][0] > 0.5 else "M")
        draw_label(frame, (face[0], face[1]), label)
    frame = cv2.resize(frame,(1920,1080),interpolation = cv2.INTER_CUBIC)
    cv2.imshow('Keras Faces', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()


