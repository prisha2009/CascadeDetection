import cv2 as cv

# pedestrians = cv.CascadeClassifier("hogcascade_pedestrians.xml")
# fourWheel = cv.CascadeClassifier("fourWheeler.xml")
sixWheel = cv.CascadeClassifier("sixWheeler.xml")
# twoWheel = cv.CascadeClassifier("twoWheeler.xml")
video = cv.VideoCapture("video (16).mp4")


while True:


    frame,image = video.read()
    grey = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    # person = pedestrians.detectMultiScale(grey, 1.3,2)
    # fourWheeler = fourWheel.detectMultiScale(grey, 1.1,2)
    sixWheeler = sixWheel.detectMultiScale(grey, 1.16,1)
    # twoWheeler = twoWheel.detectMultiScale(grey, 1.19,1)

    
    # print(person)

    # for (x,y,w,h) in person:
    #     cv.rectangle(image, (x,y), (x+w,y+h), (0,0,255), 10)
    # for (x2,y2,w2,h2) in fourWheeler:
    #     cv.rectangle(image, (x2,y2), (x2+w2,y2+h2), (255,0,0), 10)
    for (x3,y3,w3,h3) in sixWheeler:
        cv.rectangle(image, (x3,y3), (x3+w3,y3+h3), (0,255,0), 10)
    # for (x4,y4,w4,h4) in twoWheeler:
    #     cv.rectangle(image, (x4,y4), (x4+w4,y4+h4), (255,0,255), 10)


    cv.imshow("pedestrians", image)
    key = cv.waitKey(1)
    if key == ord("q"):
        break

video.release()
cv.destroyAllWindows()