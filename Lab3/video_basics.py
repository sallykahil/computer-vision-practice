import cv2
cam=cv2.VideoCapture(0)

frame_width=int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height=int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

import time
time.sleep(2.0)

#infifnite loop to capture video frames
while True:
    ret,frame=cam.read()
    
    if not ret:
        print("error in retrieving frame")
        break
    if frame.shape[0]>0 and frame.shape[1]>0:
        cv2.imshow("Video Frame",frame)
    else:
        print("Invalid dimensions")
        break 
    if cv2.waitKey(1)==ord("q"):
        break

cam.release()
cv2.destroyAllWindows()

#here we open the camera 