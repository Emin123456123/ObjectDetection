import cv2
import numpy as np

#set input image and show it
img_src = cv2.imread("E:/Open CV mini projekat/opencv/input/test04.jpg")
cv2.imshow('input',  img_src)
cv2.waitKey(1)

#get a list of recognizable objects
classes_names = open('E:/Open CV mini projekat/opencv/yolov3/coco.names').read().split('\n')

#create a list of colors for each recongizable object to use for the detection bounding rectangle
np.random.seed(42)
colors_rnd = np.random.randint(100, 255, size=(len(classes_names), 3), dtype='uint8')

#set a recognition model, preferred backend and execution target - CPU or GPU
net_yolo = cv2.dnn.readNetFromDarknet('E:/Open CV mini projekat/opencv/yolov3/yolov.cfg', 'E:/Open CV mini projekat/opencv/yolov3/yolov.weights')
net_yolo.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net_yolo.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

#get all of the network layers
ln = net_yolo.getLayerNames()
ln = [ln[i - 1] for i in net_yolo.getUnconnectedOutLayers()]

#opencv needs a blob
blob_img = cv2.dnn.blobFromImage(img_src, 1/255.0, (416, 416), swapRB=True, crop=False)
r_blob = blob_img[0, 0, :, :]

#show the blob
cv2.imshow('blob', r_blob)
cv2.waitKey(1)
text = f'Blob shape={blob_img.shape}'

#pass the blob to the network
net_yolo.setInput(blob_img)
outputs = net_yolo.forward(ln)


#run analysis and display detected objects as bounding rectangles, names and confidence
boxes = []
confidences = []
classIDs = []
h, w = img_src.shape[:2]

for output in outputs:
    for detection in output:
        scores_yolo = detection[5:]
        classID = np.argmax(scores_yolo)
        confidence = scores_yolo[classID]
        if confidence > 0.5: #confidence threshold - detected objects with confidence below this threshold will be dropped
            box_rect = detection[:4] * np.array([w, h, w, h])
            (centerX, centerY, width, height) = box_rect.astype("int")
            x_c = int(centerX - (width / 2))
            y_c = int(centerY - (height / 2))
            box_rect = [x_c, y_c, int(width), int(height)]
            boxes.append(box_rect)
            confidences.append(float(confidence))
            classIDs.append(classID)

indices_yolo = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
if len(indices_yolo) > 0:
    for i in indices_yolo.flatten():
        (x, y) = (boxes[i][0], boxes[i][1])
        (w, h) = (boxes[i][2], boxes[i][3])
        color = [int(c) for c in colors_rnd[classIDs[i]]]
        cv2.rectangle(img_src, (x, y), (x + w, y + h), color, 3)
        text = "{}: {:.4f}".format(classes_names[classIDs[i]], confidences[i])
        print(text)
        cv2.putText(img_src, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

cv2.imwrite('E:/Open CV mini projekat/opencv/output/test04detect_v3.jpg',img_src)
cv2.imshow('output', img_src)
cv2.waitKey(0)
cv2.destroyAllWindows()