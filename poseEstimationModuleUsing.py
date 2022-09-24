import cv2
import mediapipe as mp
import time
import poseEstimationModule as pem

video = "Your local video address"

def main():
    cap = cv2.VideoCapture(video)
    cTime = 0
    pTime = 0
    detector = pem.poseDetector()
    while True:
        success, img = cap.read()
        img = detector.findPose(img)
        lmList = detector.findPosition(img)
        if len(lmList) != 0:
            print(lmList[0])

        cTime = time.time()
        if  cTime-pTime != 0:
            fps = 1 / (cTime-pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()
