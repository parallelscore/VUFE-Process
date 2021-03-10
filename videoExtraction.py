import cv2
import sys

cascPath = "face_detector.xml"
faceCascade = cv2.CascadeClassifier(cascPath)


video_capture = cv2.VideoCapture("TestVideo.mp4")



while (video_capture.isOpened()):
    
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 4)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    cv2.imwrite('./SavedImages/Video.png', frame)
    print('Faces was detected & snapshot saved sucessfully')
 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


video_capture.release()
cv2.destroyAllWindows()
