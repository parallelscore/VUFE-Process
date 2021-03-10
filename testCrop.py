# import cv2
# import sys

# #imagePath = sys.argv[1]
# cascPath = "face_detector.xml"
# faceCascade = cv2.CascadeClassifier(cascPath)

# #can also provide a video file here and python would read and capture using ffmpeg
# #defauly webcam
# video_capture = cv2.VideoCapture("TestVideo.mp4")



# while (video_capture.isOpened()):
#     # Capture frame-by-frame ret-is for when we run out of frames but since it's webcam we can record forever
#     ret, frame = video_capture.read()

#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     faces = faceCascade.detectMultiScale(
#         gray,
#         scaleFactor=1.1,
#         minNeighbors=5,
#         minSize=(30, 30),
#         #flags=cv2.cv.CV_HAAR_SCALE_IMAGE
#     )

#     # Draw a rectangle around the faces
#     # for (x, y, w, h) in faces:
#     #     cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 4)

# for f in faces:

#         x, y, w, h = [ v for v in f ]
        
#         cv2.rectangle(image_copy, (x,y), (x+w, y+h), (255,0,0), 3)
#         # Define the region of interest in the image  
#         face_crop.append(gray[y:y+h, x:x+w])

# for face in face_crop:
#         cv2.imshow('face',face)
#         cv2.waitKey(0)

#         face_crop = gray[y:y+h, x:x+w]

#         # Display the image with the bounding boxes
#         fig = plt.figure(figsize = (9,9))
#         axl = fig.add_subplot(111)
#         axl.set_xticks([])
#         axl.set_yticks([])

#         ax1.set_title("Obamas with Face Detection")
#         axl.imshow(image_copy)

#         # Display the face crops
#         fig = plt.figure(figsize = (9,9))
#         axl = fig.add_subplot(111)
#         axl.set_xticks([])
#         axl.set_yticks([])

#         axl.set_title("Obamas Face Crops")
#         axl.imshow(face_crop)



#     # Display the resulting frame
# cv2.imshow('Video', frame)

# cv2.imwrite('Video.png', frame)
# print('Face was detected & snapshot saved sucessfully')
# #searching for frame - q key exists the script

#  if cv2.waitKey(1) & 0xFF == ord('q'):
#     break

# #     cv2.imwrite("face_detected.mp4", frame) 
# #  print('Face was detected sucessfuly saved')

# # When everything is done, release the capture


# video_capture.release()
# cv2.destroyAllWindows()
