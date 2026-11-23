
from picamera2 import Picamera2
import cv2
import time

# Initialise camera
picam2 = Picamera2()

config = picam2.create_preview_configuration(
    main={"size": (640, 480), "format": "RGB888"}
)

picam2.configure(config)
picam2.start()

time.sleep(2)

# Initialise ArUco detector
aruco = cv2.aruco

dictionary = aruco.getPredefinedDictionary(
    aruco.DICT_4X4_50
)

parameters = aruco.DetectorParameters()
detector = aruco.ArucoDetector(dictionary, parameters)

print("ArUco detection started")

try:
    while True:
        # Capture image
        frame = picam2.capture_array()

        # Convert RGB to grayscale
        gray = cv2.cvtColor(
            frame, cv2.COLOR_RGB2GRAY
        )

        # Detect marker
        corners, ids, rejected = detector.detectMarkers(gray)

        if ids is not None:
            aruco.drawDetectedMarkers(frame, corners, ids)

            for marker_id, corner in zip(ids.flatten(), corners):
                center = corner[0].mean(axis=0)

                print(
                    f"Marker ID: {marker_id}, "
                    f"Center: {center.astype(int)}"
                )

        # Display camera feed
        cv2.imshow("ArUco Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    picam2.stop()
    cv2.destroyAllWindows()
