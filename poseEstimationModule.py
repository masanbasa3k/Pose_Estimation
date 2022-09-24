import cv2
import mediapipe as mp
import time

video = "Your local video address"

class poseDetector():
    def __init__(self, mode = False, upperBody = False, smooth = True, detectioncon = 0.5, trackcon = 0.5):
        self.mode = mode
        self.upperBody = upperBody
        self.smooth = smooth
        self.detectioncon = detectioncon
        self.trackcon = trackcon

        
        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode,self.upperBody,self.smooth,self.detectioncon,self.trackcon)

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        return img

    def findPosition(self, img, draw=True):
        lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h) 
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255,0,0), cv2.FILLED)
        return lmList
    

def main():
    cap = cv2.VideoCapture(video)
    cTime = 0
    pTime = 0
    detector = poseDetector()
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

        cv2.putText(img, "FPS "+str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()
