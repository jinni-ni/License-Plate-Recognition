import cv2
import numpy as np

def detectAndDisplay(frame):
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.equalizeHist(frame_gray)

    #-- Detect faces
    licenses = license_plate.detectMultiScale(frame_gray)
    
    for (x,y,w,h) in licenses:
        center = (x + w//2, y + h//2)
        frame = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 4)
        licenseROI = frame_gray[y:y+h,x:x+w]
      
            
    cv2.imshow('Capture - License detection', frame)        

print("version:")
print(cv2.__version__)

img = cv2.imread("../image/test02.jpg")

print("width: {} pixcels".format(img.shape[1]))
print("height: {} pixcels".format(img.shape[0]))
print("channels; {}".format(img.shape[2]))


(height, width) = img.shape[:2]

cv2.imshow("original Image", img)

license_plate_cascade = '../haarcascades/haarcascade_russian_plate_number.xml'
license_plate = cv2.CascadeClassifier()


if not license_plate.load(cv2.samples.findFile(license_plate_cascade)):
    print('--(!)Error loading license plate')
    exit(0)    

detectAndDisplay(img)

cv2.waitKey(0)
cv2.destroyAllWindows()
