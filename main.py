import cv2 as cv
import math
# import pandas


ageList = ["0-2","4-6","8-12","13-17","19-23","25-30","30-35","35-40","42-47","50-60","60-70","70-80"]
genderList = ["male","female","null"]
# img = cv.imread("face.jpg")
agePrototype = "age_deploy.prototxt"
facePrototype = "opencv_face_detector_uint8.pb"
ageModel = "age_net.caffemodel"
genderPrototype = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"
faceModel = "opencv_face_detector.pbtxt"


def highlightFace(net,a,threshold = 0.7):
    a = a.copy()
    frameHeight = a.shape[0]
    frameWidth = a.shape[1]

    blob = cv.dnn.blobFromImage(a,1.0,(200,200),[67,48,120],True,False)
    net.setInput(blob)
    faceboxes = []
    directions = net.forward()
    print(directions.shape[2])

    for i in range(directions.shape[2]):
        confidence = directions[0,0,i,2]            # directions[x,z,-x,-z]
        if confidence > threshold:
            x1 = directions[0,0,i,3] * frameWidth
            y1 = directions[0,0,i,4] * frameHeight
            x2 = directions[0,0,i,5] * frameWidth
            y2 = directions[0,0,i,6] * frameHeight
            faceboxes.append([x1,y1,x2,y2])
            # cv2.rectangle()

    return a, faceboxes

video = cv.VideoCapture(0)

faceNet = cv.dnn.readNet(faceModel,facePrototype)
ageNet = cv.dnn.readNet(ageModel,agePrototype)
genderNet = cv.dnn.readNet(genderModel,genderPrototype)

coordinates = (70,80,100)

while True:
    global face
    x, frame = video.read()
    if not x:
        cv.waitKey()
        break
    resultImage, facebox = highlightFace(faceNet,frame)

    # if not resultImage:
    #     cv.waitKey()
    #     break
    
    print(facebox)

    for box in facebox:
        face = frame[max(0,box[1]-20):
        min(box[3]+20,frame.shape[0]-1),max(0,box[0]-20)
        :min(box[2]+20,frame.shape[1]-1)]
    
        blob = cv.dnn.blobFromImage(face,1.0,(200,200),coordinates,swapRB = False)
        genderNet.setInput(blob)
        genderD = genderNet.forward()
        gender = genderList[genderD[0].argmax()]
        # print(genderD)

        ageNet.setInput(blob)
        ageD = ageNet.forward()
        age = ageList[ageD[0].argmax()]
        # print(ageD)

        cv.imshow("Face",resultImage)
        cv.putText(resultImage,f"{gender},{age}" , (facebox[0],facebox[1]) , cv.FONT_HERSHEY_COMPLEX, 1 , (255,0,0),2,cv.LINE_AA )

    if cv.waitKey(1) == ord("q"):
        break

cv.destroyAllWindows()