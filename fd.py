import cv2
import sys
import math
import numpy as np
from numpy.linalg import inv
# Get user supplied values
# imagePath = sys.argv[1]
# cascPath = sys.argv[2]



def dis(p0, p1): 
    x = p1[0]-p0[0]
    y = p1[1]-p0[1]
    return math.sqrt(x**2+y**2)


def Matx(ang):
    M = np.float32([[math.cos(ang), -math.sin(ang), 0],
        [math.sin(ang), math.cos(ang), 0]])
    return M 


def direction(eyes):
    return [eyes[0][1] - eyes[1][1], eyes[0][0] - eyes[1][0]]






# Create the haar cascade

#faceCascade = cv2.CascadeClassifier(cascPath)
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
nested = cv2.CascadeClassifier('haarcascade_eye.xml')
# Read the image
# image = cv2.imread(imagePath)
# image = cv2.imread('abba.png') WP_20150523_005.jpg

image = cv2.imread('')   # import image here 
# image = cv2.imread('ema.jpg')
image = cv2.pyrDown(image)       #downsample befor grayscaleing 
image = cv2.pyrDown(image)
cpimg = image
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30),
    flags = cv2.cv.CV_HAAR_SCALE_IMAGE
)

eyes = nested.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30),
    flags = cv2.cv.CV_HAAR_SCALE_IMAGE
)

print "Found {0} faces!".format(len(faces))

# Draw a rectangle around the faces
for (a, b, c, d) in faces:
    # cv2.rectangle(image, (a, b), (a+c, b+d), (0, 255, 0), 2)
    # roi = gray[b:b+d, a:a+c]
    # cv2.imshow("roi", roi)
    for (x, y, w, h) in eyes:
        # cv2.rectangle(image, (x, y), (x+w/2, y+h/2), (0, 255, 0), 2)
        drc = direction(eyes)
        theta = -math.atan2(drc[0],drc[1])
        middis = dis(eyes[1],eyes[0]) / 2


midy =  np.float32(-middis*math.sin(theta) + y+h/2)
midx =  np.float32(middis*math.cos(theta) + x+w/2)
mid = (midx,midy,0)
midtran = np.dot(Matx(theta), mid)     
midtran = np.float32((midtran[0], midtran[1]))     #transaltion of midpoint of eyes
for n in range(0,len(midtran)):
    midtran[n] = math.floor(midtran[n])


midtran = (midtran[0], midtran[1])
rows,cols,junk = cpimg.shape
cpimg = cv2.warpAffine(image, Matx(theta), (cols,rows))
# cv2.circle(cpimg,midtran,10, (0,255,0), 2)

facetran = (faces[0][0], faces[0][1], 0)
facetran = np.dot(Matx(theta),facetran)
facetr = []
for n in range(0,len(facetran)):
    facetr.append(int(facetran[n]))



eyeDisX = middis*2
faceDisX = ((eyeDisX*279)/138)


roi = cpimg[facetr[1]:facetr[1]+d,midtran[0]-(faceDisX/2):midtran[0]+(faceDisX/2)]
# cv2.rectangle(cpimg, (facetr[0], facetr[1]), (facetr[0]+c, facetr[1]+d), (0, 255, 0), 2)


cv2.imshow("cp", cpimg)
# rows,cols = roi.shape
# a = cv2.warpAffine(roi, Matx(theta) , (cols,rows))
# cv2.imshow("a", a)
# cv2.circle(a,midtran, 10, (0,255,0), 2)


# cv2.imshow("roi not resized", roi)
# roi = cv2.resize(roi, (112,112))
roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
cv2.imshow("roi resized", roi)
# cv2.imwrite('', roi)








cv2.imshow("Faces found", image)

cv2.waitKey(0)





