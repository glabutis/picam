import cv2
import imutils
import time
from imutils.video import VideoStream
import numpy as np
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True, help="path to Caffe")
ap.add_argument("-m", "--model", required=True, help="path to model")
ap.add_argument("-c", "--confidence", type=float, default=0.5, help="min probably to help detect weak detectionsssss")
args = vars(ap.parse_args())

print("[Info] Stream starting soon")

vs = VideoStream(src=0).start()
time.sleep(2.0)



net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=400)

    #this is the blob part
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, 
    (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    #pass blob through net
    net.setInput(blob)
    detections = net.forward()

    #loop over detections 
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence < args["confidence"]:
            continue
        
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        #draw the fancy box
        text = "{:.2f}%".format(confidence * 100)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(frame, (startX, startY), (endX, endY),
        (0, 0, 255), 2)
        cv2.putText(frame, text, (startX, startY),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break