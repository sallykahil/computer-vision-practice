import cv2 as cv
import numpy as np

face_cascade=cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade=cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_eye.xml')

img=cv.imread('../data/nadia_murad.jpg')
img_grey=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
faces=face_cascade.detectMultiScale(img_grey,1.3,5)

print(f"Number of faces detected: {len(faces)}")
print(faces[0])  #print the first face coordinates
for(x,y,w,h) in faces:
    cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

    #region of interest for eyes within the detected face
    roi_gray=img_grey[y:y+h,x:x+w]
    roi_color=img[y:y+h,x:x+w]

    eyes=eye_cascade.detectMultiScale(roi_gray,2.9)
    for(ex,ey,ew,eh) in eyes:
        cv.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

cv.imshow('Detected Face',img)
cv.waitKey(0)
cv.destroyAllWindows()