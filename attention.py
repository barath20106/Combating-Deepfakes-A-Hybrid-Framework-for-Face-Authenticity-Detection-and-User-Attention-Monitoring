import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh


class AttentionTracker:
    def __init__(self):
        self.calibrated = False
        self.neutral_pitch = 0
        self.neutral_yaw = 0
        self.neutral_eye = 0
        self.PITCH_THRESHOLD = 0.3
        self.YAW_THRESHOLD = 0.3
        self.EYE_THRESHOLD = 0.15

    def calibrate(self, frame, samples=30):
        h, w = frame.shape[:2]
        face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

        neutral_pitch_list, neutral_yaw_list, neutral_eye_list = [], [], []

        for _ in range(samples):
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = face_mesh.process(rgb_frame)

            if result.multi_face_landmarks:
                landmarks = result.multi_face_landmarks[0]
                landmark_ids = [1, 33, 263, 61, 291, 199]
                pts = np.array([
                    (landmarks.landmark[i].x * w, landmarks.landmark[i].y * h)
                    for i in landmark_ids
                ], dtype=np.float32)

                model_pts = np.array([
                    (0.0, 0.0, 0.0),
                    (-30.0, -125.0, -30.0),
                    (-60.0, 40.0, -60.0),
                    (60.0, 40.0, -60.0),
                    (-40.0, -50.0, -60.0),
                    (40.0, -50.0, -60.0)
                ], dtype=np.float32)

                cam_matrix = np.array([[w, 0, w / 2],
                                       [0, w, h / 2],
                                       [0, 0, 1]], dtype="double")
                dist_coeffs = np.zeros((4, 1))

                success, rotation_vector, _ = cv2.solvePnP(model_pts, pts, cam_matrix, dist_coeffs)

                if success:
                    rotation_vector = rotation_vector.flatten()
                    neutral_pitch_list.append(rotation_vector[0])
                    neutral_yaw_list.append(rotation_vector[1])

                    left_eye_x = landmarks.landmark[33].x
                    right_eye_x = landmarks.landmark[263].x
                    neutral_eye_list.append((left_eye_x + right_eye_x) / 2 - 0.5)

        if neutral_pitch_list:
            self.neutral_pitch = np.mean(neutral_pitch_list)
            self.neutral_yaw = np.mean(neutral_yaw_list)
            self.neutral_eye = np.mean(neutral_eye_list)
            self.PITCH_THRESHOLD = 0.6
            self.YAW_THRESHOLD = 0.6
            self.EYE_THRESHOLD = 0.2
            self.calibrated = True

    def get_attention(self, frame):
        h, w = frame.shape[:2]
        face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb_frame)

        attention_status = "No Face Detected"
        head_pose = (0, 0, 0)

        if result.multi_face_landmarks:
            landmarks = result.multi_face_landmarks[0]
            landmark_ids = [1, 33, 263, 61, 291, 199]

            pts = np.array([
                (landmarks.landmark[i].x * w, landmarks.landmark[i].y * h)
                for i in landmark_ids
            ], dtype=np.float32)

            model_pts = np.array([
                (0.0, 0.0, 0.0),
                (-30.0, -125.0, -30.0),
                (-60.0, 40.0, -60.0),
                (60.0, 40.0, -60.0),
                (-40.0, -50.0, -60.0),
                (40.0, -50.0, -60.0)
            ], dtype=np.float32)

            cam_matrix = np.array([[w, 0, w / 2],
                                   [0, w, h / 2],
                                   [0, 0, 1]], dtype="double")
            dist_coeffs = np.zeros((4, 1))

            success, rotation_vector, _ = cv2.solvePnP(model_pts, pts, cam_matrix, dist_coeffs)

            if success:
                head_pose = rotation_vector.flatten()
                left_eye_x = landmarks.landmark[33].x
                right_eye_x = landmarks.landmark[263].x
                eye_dir = (left_eye_x + right_eye_x) / 2 - 0.5

                if not self.calibrated:
                    attention_status = "Not Calibrated"
                else:
                    pitch_ok = abs(head_pose[0] - self.neutral_pitch) < self.PITCH_THRESHOLD
                    yaw_ok = abs(head_pose[1] - self.neutral_yaw) < self.YAW_THRESHOLD
                    eye_ok = abs(eye_dir - self.neutral_eye) < self.EYE_THRESHOLD

                    attention_status = "Paying Attention" if (pitch_ok and yaw_ok and eye_ok) else "Distracted"

        return attention_status, head_pose
