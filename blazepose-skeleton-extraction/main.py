import cv2
import mediapipe as mp
import time

mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose(
    min_detection_confidence=0.65,
    min_tracking_confidence=0.65)

cap = cv2.VideoCapture('PoseVideos/demo1.mp4')
pTime = 0

while True:
    ### Read and Resize video image
    success, img = cap.read()
    img = cv2.resize(img, (960, 540))
    ##img = cv2.resize(img, (405, 768)) ## for demo2

    ### Convert to RGB
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
    ## => Framerate decrease

    ### Draw skeleton:
    print(results.pose_landmarks)
    if results.pose_landmarks:
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)

    ### Time capture
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN,3, (255,0,0), 3)
    cv2.imshow("Image", img)

    ### Delay
    cv2.waitKey(10)
