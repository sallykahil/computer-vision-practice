import numpy as np
import cv2 
import imutils 

def align_images(image, template, maxFeatures=500, keepPercent=0.2, debug=False):
    #first convert the images to grayscale
    imageGray=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    templateGray=cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    #detect feaatures and compute the descriptors using ORB

    orb=cv2.ORB_create(maxFeatures)
    (kpsA,descA)=orb.detectAndCompute(imageGray,None)
    (kpsB,descB)=orb.detectAndCompute(templateGray,None)

    #match the features using the bruteforce matcher
    method=cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING
    matcher=cv2.DescriptorMatcher_create(method)
    matches=matcher.match(descA,descB)

    #sort the matches by their distance
    matches=sorted(matches,key=lambda x:x.distance)
    '''' Explanation of sorted function:
sorted(...) creates a new sorted list
key=lambda x: x.distance tells Python:
“Sort each element using its distance value”
Matches are ordered from smallest distance to largest
Result
Best matches come first
Worst matches come last
sorted=[x1,x2]
for x in sorted:
    print(x.distance)
    IF x1.distance < x2.distance:
        x1 comes before x2 in the sorted list
    '''
    #keep only the top matches based on the keepPercent
    keep=int(len(matches)*keepPercent)
    matches=matches[:keep]

    ptsA=np.zeros((len(matches),2),dtype="float")
    ptsB=np.zeros((len(matches),2),dtype="float")
    if debug:
        matchedVis=cv2.drawMatches(image,kpsA,template,kpsB,matches,None)
        matchedVis=imutils.resize(matchedVis,width=1000)
        cv2.imshow("Matched Keypoints",matchedVis)
        cv2.waitKey(0)

  # Populate the point arrays 
  #enyumerate function adds a counter to an iterable and returns it as an enumerate object
    for(i,m) in enumerate(matches):
        ptsA[i]=kpsA[m.queryIdx].pt
        ptsB[i]=kpsB[m.trainIdx].pt

    #compute the homography matrix to align the images
    (H,mask)=cv2.findHomography(ptsA,ptsB,cv2.RANSAC)
    #use the homography matrix to warp the images
    (h,w)=template.shape[:2]
    aligned=cv2.warpPerspective(image,H,(w,h))
    return aligned
image=cv2.imread("image.jpg")
template=cv2.imread("main.png")
print("Aligning images...")
aligned=align_images(image,template,debug=True)


aligned=imutils.resize(aligned,width=700)
template=imutils.resize(template,width=700)
stacked=np.hstack([template,aligned])
cv2.imshow("stacked Images",stacked)

overlay=template.copy()
output=template.copy()
cv2.addWeighted(aligned,0.5,overlay,0.5,0,output)
cv2.imshow("Overlayed Images",output)        
cv2.waitKey(0)