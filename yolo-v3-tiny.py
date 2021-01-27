import cv2
import numpy as np

min_confidence = 0.3

# Load Yolo
net = cv2.dnn.readNet("./yolo-tiny/yolov3-tiny.weights", "./yolo-tiny/yolov3-tiny.cfg")
classes = []

# Blur list
# 0 person 2 car 3 motorbike 5 bus 7 truck
b_list = [0, 2, 3, 5, 7]

with open("./yolo-tiny/coco.names","r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes),3))

img = cv2.imread("./py_image/test03.png")
#img = cv2.resize(img,None, fx=0.4, fy=0.4)
height, width, channels = img.shape

blob = cv2.dnn.blobFromImage(img, 0.00392, (416,416), (0,0,0), True, crop=False)

net.setInput(blob)
outs = net.forward(output_layers)

class_ids = []
confidences = []
boxes = []

for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]

        #if class_id not in b_list:
        #    continue
        if confidence > min_confidence:
            # Object detected
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            # Rectangle coordinates
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)
            
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)


indexes = cv2.dnn.NMSBoxes(boxes, confidences, min_confidence, 0.4)
font = cv2.FONT_HERSHEY_PLAIN
for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        print(class_ids[i])
        print(i, label)
        color = colors[i]
        #roi = img[y:y+h, x:x+w]
        #roi = cv2.blur(roi, (9,9))
        #img[y:y+h, x:x+w] = roi
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, label, (x, y + 30), font, 2, (0, 255, 0), 1)
        

cv2.imshow("YOLO Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()    
    

        
