import cv2 as cv

cascade = cv.CascadeClassifier("frontal_face.xml")
eye = cv.CascadeClassifier("harcascade_eye.xml")
smile = cv.CascadeClassifier("harcascade_smile.xml")

video = cv.VideoCapture(0)

while True:
    frame, image = video.read() # frame is the information about the video and image is video itself
    grey = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(grey,1.3,5)
    for (x,y,w,h) in faces:
        # cv.circle(img,(x,y),5,(0,0,255),2)
        cv.rectangle(image,(x,y),(x+w,y+h),(0,0,255),2)


        smileGrey = grey[y:y+h,x:x+w]
        smileImage = image[y:y+h,x:x+w]
        smiles = smile.detectMultiScale(smileGrey,1.8,20)
        print(smiles)

        for(sx,sy,sw,sh) in smiles:
            cv.rectangle(smileImage, (sx,sy), (sx+sw,sy+sh), (125,0,0), 2)


    cv.imshow('face',image)
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break

video.release()
cv.destroyAllWindows()