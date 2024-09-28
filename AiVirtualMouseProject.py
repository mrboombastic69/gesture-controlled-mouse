import cv2
import numpy as np
import HandTrackingModule as htm
import time
import autopy

# Set webcam dimensions
webcamWidth, webcamHeight = 640, 480
# Frame reduction for mouse movement area
frameReduction = 100
# Smoothening factor to control mouse movement
smootheningFactor = 7

# Previous and current time for calculating frame rate
previousTime = 0
# Previous and current location of the mouse pointer
previousLocationX, previousLocationY = 0, 0
currentLocationX, currentLocationY = 0, 0

# Initialize webcam capture
capture = cv2.VideoCapture(0)
capture.set(3, webcamWidth)
capture.set(4, webcamHeight)
# Initialize hand detector with a maximum of one hand
detector = htm.HandDetector(maxHands=1)
# Get the screen size
screenWidth, screenHeight = autopy.screen.size()

while True:
    # 1. Find hand landmarks
    success, image = capture.read()
    if not success:
        break
    image = detector.findHands(image)
    landmarksList, boundingBox = detector.findPosition(image)

    # 2. Get the tip positions of the index, middle, and pinky fingers
    if len(landmarksList) != 0:
        indexFingerX, indexFingerY = landmarksList[8][1:]  # Index finger tip
        middleFingerX, middleFingerY = landmarksList[12][1:]  # Middle finger tip
        pinkyFingerX, pinkyFingerY = landmarksList[20][1:]  # Pinky finger tip

        # 3. Check which fingers are up
        fingersStatus = detector.fingersUp()

        # Debug: Print finger states
        print(f"Fingers Up: {fingersStatus}")

        # 4. Only Index Finger Up: Moving Mode
        if fingersStatus[1] == 1 and all(f == 0 for i, f in enumerate(fingersStatus) if i != 1):
            # 5. Convert coordinates to screen size
            screenX = np.interp(indexFingerX, (frameReduction, webcamWidth - frameReduction), (0, screenWidth))
            screenY = np.interp(indexFingerY, (frameReduction, webcamHeight - frameReduction), (0, screenHeight))
            # 6. Smoothen the values
            currentLocationX = previousLocationX + (screenX - previousLocationX) / smootheningFactor
            currentLocationY = previousLocationY + (screenY - previousLocationY) / smootheningFactor
            # 7. Move the mouse to the calculated coordinates
            autopy.mouse.move(screenWidth - currentLocationX, currentLocationY)
            cv2.circle(image, (indexFingerX, indexFingerY), 15, (255, 0, 255), cv2.FILLED)
            previousLocationX, previousLocationY = currentLocationX, currentLocationY

        # 8. Left Click: Index and Middle Fingers Up
        if fingersStatus[1] == 1 and fingersStatus[2] == 1 and all(f == 0 for i, f in enumerate(fingersStatus) if i not in [1, 2]):
            length, image, lineInfo = detector.findDistance(8, 12, image)
            if length < 40:
                autopy.mouse.click(autopy.mouse.Button.LEFT)

        # 9. Right Click: Index and Pinky Fingers Up
        if fingersStatus[1] == 1 and fingersStatus[4] == 1 and all(f == 0 for i, f in enumerate(fingersStatus) if i not in [1, 4]):
            length, image, lineInfo = detector.findDistance(8, 20, image)

            # Debug: Print distance between index and pinky fingers
            print(f"Right Click Distance: {length}")

            # Set a threshold distance for the right-click gesture
            thresholdDistance = 100  # You can adjust this value based on your preference
            if length > thresholdDistance:
                print("Right Click Triggered")
                autopy.mouse.click(autopy.mouse.Button.RIGHT)

    # 10. Calculate the frame rate
    currentTime = time.time()
    framesPerSecond = 1 / (currentTime - previousTime)
    previousTime = currentTime
    cv2.putText(image, str(int(framesPerSecond)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    # 11. Display the image
    cv2.imshow("Image", image)
    cv2.waitKey(1)
