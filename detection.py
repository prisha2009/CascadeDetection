import cv2 as cv

cascade = cv.CascadeClassifier("frontal_face.xml")
# img = cv.imread("faces.jpg")    
# gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# faces = cascade.detectMultiScale(gray,1.3,5)
# print(faces)          # [x,y,width,height]

# for (x,y,w,h) in faces:
#     # cv.circle(img,(x,y),5,(0,0,255),2)
#     cv.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)

# cv.imshow('face',img)
# cv.waitKey(0)

video = cv.VideoCapture(0)

while True:
    frame, image = video.read()
    grey = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(grey,1.3,5)
    for (x,y,w,h) in faces:
        # cv.circle(img,(x,y),5,(0,0,255),2)
        cv.rectangle(image,(x,y),(x+w,y+h),(0,0,255),2)
    
    cv.imshow('face',image)
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break

video.release()
cv.destroyAllWindows()