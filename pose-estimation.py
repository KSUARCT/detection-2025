import cv2
import apriltag
from imutils.video import VideoStream
import time
from imutils.video import FPS
import numpy as np
from pymavlink.dialects.v20 import common as mavlink2
from pymavlink import mavutil

from tag_processor import TagProcessor


# Camera calibration parameters (intrinsic parameters)
fx = 1.0 * 1067  # Focal length in x direction
fy = 1.0 * 1067  # Focal length in y direction
cx = 1.0 * 640  # Optical center in x direction
cy = 1.0 * 360  # Optical center in y direction
camera_matrix = np.array([[fx, 0, cx],
                          [0, fy, cy],
                          [0,  0,  1]])
k1, k2, p1, p2, k3 = 0, 0, 0, 0, 0  # Distortion coefficients
dist_coeffs = np.zeros([k1, k2, p1, p2, k3])  # Assuming no distortion for simplicity


serial_port = "/dev/ttyACM0"
baud_rate = 57600
conn = mavutil.mavlink_connection(
    serial_port,
    baud=baud_rate,
    source_system=1,
    source_component=1,
    autoreconnect=True,
    force_connected=True,
    retries=3,
    timeout=0.5
)

conn.wait_heartbeat()
print("Heartbeat from system (system %u component %u)" % (conn.target_system, conn.target_component))

# Display a confirmation message in mission planner through the flight controller
conn.mav.statustext_send(mavlink2.MAV_SEVERITY_INFO, "AprilTag detection started".encode('utf-8'))

mavframe = mavlink2.MAV_FRAME_BODY_FRD
target_type = mavlink2.LANDING_TARGET_TYPE_VISION_FIDUCIAL

# Initialize the AprilTag detector
config = apriltag.DetectorOptions(families="tag36h11")
detector = apriltag.Detector(config)


tag_processor = TagProcessor(camera_matrix, dist_coeffs, tag_size=0.05)


# Setup video stream and FPS counter
vs = VideoStream(src=0).start()
time.sleep(2.0)
fps = FPS().start()

while True:
    frame = vs.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    results = detector.detect(gray)

    success, frame, rvec, tvec = tag_processor.process(frame, results)

    if success:
        print(f"Rotation vector: {rvec}")
        print(f"Translation vector: {tvec}")

    # Display the frame with detected tags
    cv2.imshow("AprilTag Detection", frame)

    # Send the Target Position and Orientation to the flight controller using Landing Target Protocol
    if success:
        # Convert the rotation vector to degrees
        rdeg = rvec * 180 / np.pi
        x = tvec[0][0]
        y = tvec[1][0]
        z = tvec[2][0]
        q = [1, rvec[0][0], rvec[1][0], rvec[2][0]]

        # Send the target position and orientation to the flight controller
        conn.mav.landing_target_send(
            mavframe,  # MAV_FRAME
            x,  # X-axis
            y,  # Y-axis
            z,  # Z-axis
            q,  # Quaternion
            target_type,  # Type of landing target
            1,  # Position valid
        )


    # Check if the user pressed the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Update the FPS counter
    fps.update()

cv2.destroyAllWindows()
vs.stop()
fps.stop()
print("Elapsed time: {:.2f}".format(fps.elapsed()))
print("Approx. FPS: {:.2f}".format(fps.fps()))
