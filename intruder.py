#importing libraries
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import imutils
import time
import cv2
# import notification

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

#we'll import library for notification till then let the func passs
# def notification():
#     pass


print("[INFO] loading model...")
#models data is called.
net = cv2.dnn.readNetFromCaffe('mobilenet_ssd/MobileNetSSD_deploy.prototxt',
                               'mobilenet_ssd/MobileNetSSD_deploy.caffemodel')

facenet = cv2.dnn.readNetFromCaffe('deploy.prototxt.txt', 'res10_300x300_ssd_iter_140000.caffemodel')


vs = VideoStream(src=0).start()
time.sleep(2.0)


W = None
H = None

fps = FPS().start()

while True:
    # grab the next frame and handle if we are reading from either
    # VideoCapture or VideoStream
    frame = vs.read()

    # if we are viewing a video and we did not grab a frame then we
    # have reached the end of the video
    if frame is None:
        break

    # resize the frame to have a maximum width of 500 pixels (the
    # less data we have, the faster we can process it), then convert
    # the frame from BGR to RGB for dlib
    frame = imutils.resize(frame, width=500)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # if the frame dimensions are empty, set them
    if W is None or H is None:
        (H, W) = frame.shape[:2]

    # convert the frame to a blob and pass the blob through the
    # network and obtain the detections
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
    net.setInput(blob)
    detections = net.forward()

    # loop over the detections
    for i in np.arange(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated
        # with the prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by requiring a minimum
        # confidence
        if confidence > 0.5:
            # extract the index of the class label from the
            # detections list
            idx = int(detections[0, 0, i, 1])

            # if the class label is not a person, ignore it
            if CLASSES[idx] != "person":
                continue

            # compute the (x, y)-coordinates of the bounding box
            # for the object
            box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
            (startX, startY, endX, endY) = box.astype("int")
            cv2.rectangle(frame, (startX, startY), (endX, endY),
                          (0, 0, 255), 2)

            roi_frame = frame[startY: endY, startX: endX]
            h, w = roi_frame.shape[:2]
            try:
                faceblob = cv2.dnn.blobFromImage(cv2.resize(roi_frame, (300, 300)), 1.0,
                                             (300, 300), (104.0, 177.0, 123.0))
                facenet.setInput(faceblob)
                detections2 = facenet.forward()

                #print(avg)
            except:
                print("Intruder")
                # notification()
                continue

            for i in range(0, detections2.shape[2]):
                # extract the confidence (i.e., probability) associated with the
                # prediction
                confidence1 = detections2[0, 0, i, 2]

                if confidence1 < 0.5:
                    continue

                # compute the (x, y)-coordinates of the bounding box for the
                # object
                box = detections2[0, 0, i, 3:7] * np.array([w, h, w, h])

                (startX, startY, endX, endY) = box.astype("int")

                # draw the bounding box of the face along with the associated
                # probability
                text = "{:.2f}%".format(confidence * 100)
                y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.rectangle(frame, (startX, startY), (endX, endY),
                              (0, 0, 255), 2)
                cv2.putText(frame, text, (startX, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

    fps.update()
    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

vs.stop()

# close any open windows
cv2.destroyAllWindows()
