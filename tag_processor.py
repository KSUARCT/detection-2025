import cv2
import numpy as np


class TagProcessor:
    def __init__(self, camera_matrix, dist_coeffs, tag_size):
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.tag_size = tag_size

        # 3D points of the tag corners in the tag's coordinate system
        self.obj_pts = np.array([
            [-tag_size / 2, -tag_size / 2, 0],
            [tag_size / 2, -tag_size / 2, 0],
            [tag_size / 2, tag_size / 2, 0],
            [-tag_size / 2, tag_size / 2, 0]
        ])

        # Draw a 3D cube to show the pose
        self.axis = np.float32([
            [-tag_size / 2, -tag_size / 2, 0], [tag_size / 2, -tag_size / 2, 0],
            [tag_size / 2, tag_size / 2, 0], [-tag_size / 2, tag_size / 2, 0],
            [-tag_size / 2, -tag_size / 2, -tag_size],
            [tag_size / 2, -tag_size / 2, -tag_size],
            [tag_size / 2, tag_size / 2, -tag_size],
            [-tag_size / 2, tag_size / 2, -tag_size]
        ]).reshape(-1, 3)

    def process(self, frame, results):
        for result in results:
            # print("Detected tag:", result.tag_id)

            # 2D points of the tag corners in the image
            img_pts = np.array(result.corners, dtype=np.float32)

            # The rvec and tvec are the rotation and translation vectors
            # Compute the pose using solvePnP
            success, rvec, tvec = cv2.solvePnP(self.obj_pts, img_pts, self.camera_matrix, self.dist_coeffs)

            if success:
                cv2.drawFrameAxes(frame, self.camera_matrix, self.dist_coeffs, rvec, tvec, 0.1)
                cv2.putText(frame, f"Position (x={tvec[0][0]:.2f}, y={tvec[1][0]:.2f}, z={tvec[2][0]:.2f})",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                rdeg = rvec * 180 / np.pi
                cv2.putText(frame, f"Rotation (x={rdeg[0][0]:.2f}, y={rdeg[1][0]:.2f}, z={rdeg[2][0]:.2f})",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                imgpts, jac = cv2.projectPoints(self.axis, rvec, tvec, self.camera_matrix, self.dist_coeffs)
                imgpts = imgpts.astype(np.int32)
                cv2.drawContours(frame, [imgpts[:4]], -1, (0, 255, 0), 2)
                cv2.drawContours(frame, [imgpts[4:]], -1, (0, 255, 0), 2)
                for i in range(4):
                    cv2.line(frame, imgpts[i][0], imgpts[i + 4][0], (0, 255, 0), 2)

                # Flip the y value
                return success, frame, rvec, tvec
            else:
                cv2.putText(frame, "Pose estimation failed", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255),
                            2)

        return False, frame, None, None
