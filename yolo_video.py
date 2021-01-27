import numpy as np
import cv2
import time

min_confidence = 0.5
width = 800
height = 0
show_ratio = 1.0
title_name = 'Custom Yolo'

net = cv2.dnn.readNet("./yolo-model/yolov3.weights", "./yolo-model/yolov3.cfg")
file_name = '../OpenCV/cv2/video/yolo_01.mp4'
classes = []

# Blur list
# 0 person 2 car 3 motorbike 5 bus 7 truck
b_list = [0, 2, 3, 5, 7]

with open("./yolo-model/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
    
color_lists = np.random.uniform(0, 255, size=(len(classes), 3))

layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

def detectAndDisplay(image):
    h, w = image.shape[:2]
    height = int(h * width / w)
    img = cv2.resize(image, (width, height))

    blob = cv2.dnn.blobFromImage(img, 0.003992, (416, 416), swapRB=True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)

    confidences = []
    names = []
    boxes = []
    colors = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if class_id not in b_list:
                continue
            if confidence > min_confidence:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w  / 2)
                y = int(center_y - h / 2)
            

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                names.append(classes[class_id])
                colors.append(color_lists[class_id])

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, min_confidence, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = '{} {:,.2%}'.format(names[i], confidences[i])
            color = colors[i]
            roi = img[y:y+h, x:x+w]
            roi = cv2.blur(roi, (9,9))
            img[y:y+h, x:x+w] = roi
#            print(i, label, x, y, w, h)
#            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
#            cv2.putText(img, label, (x, y - 10), font, 1, color, 2)
    cv2.imshow(title_name, img)

#-- 2. Read the video stream
cap = cv2.VideoCapture(file_name)

if not cap.isOpened:
    print('--(!)Error opening video capture')
    exit(0)
while True:
    ret, frame = cap.read()
    if frame is None:
        print('--(!) No captured frame -- Break!')
        break
    detectAndDisplay(frame)
    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cv2.destroyAllWindows()
