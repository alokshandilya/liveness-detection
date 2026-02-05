import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import threading
import os

# Global MediaPipe Initialization
_mesh_lock = threading.Lock()
face_landmarker = None

def get_face_landmarker():
    global face_landmarker
    if face_landmarker is None:
        model_path = os.path.join(os.path.dirname(__file__), 'face_landmarker.task')
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1,
            # Use IMAGE mode for stateless processing of chunks to avoid timestamp issues 
            # across different video files if we were to share a VIDEO-mode instance.
            running_mode=vision.RunningMode.IMAGE
        )
        face_landmarker = vision.FaceLandmarker.create_from_options(options)
        print("MediaPipe FaceLandmarker Initialized")
    return face_landmarker

# Constants for EAR
# Left eye indices: 33, 160, 158, 133, 153, 144 (P1..P6)
LEFT_EYE = [33, 160, 158, 133, 153, 144]
# Right eye indices: 362, 385, 387, 263, 373, 380
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# Constants for Head Pose (approximate 3D model points)
# Nose tip, Chin, Left Eye Left Corner, Right Eye Right Corner, Left Mouth Corner, Right Mouth Corner
NOSE_TIP = 1
CHIN = 152
LEFT_EYE_CORNER = 33
RIGHT_EYE_CORNER = 263
LEFT_MOUTH_CORNER = 61
RIGHT_MOUTH_CORNER = 291

# Generic 3D model points (in world coordinates)
GENERIC_3D_MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),             # Nose tip
    (0.0, -330.0, -65.0),        # Chin
    (-225.0, 170.0, -135.0),     # Left eye left corner
    (225.0, 170.0, -135.0),      # Right eye right corner
    (-150.0, -150.0, -125.0),    # Left Mouth corner
    (150.0, -150.0, -125.0)      # Right mouth corner
], dtype=np.float64)


def calculate_ear(landmarks, indices, img_w, img_h):
    # Retrieve coordinates
    coords = []
    for idx in indices:
        lm = landmarks[idx]
        coords.append(np.array([lm.x * img_w, lm.y * img_h]))
    
    # Calculate distances
    # P2-P6
    d_p2_p6 = np.linalg.norm(coords[1] - coords[5])
    # P3-P5
    d_p3_p5 = np.linalg.norm(coords[2] - coords[4])
    # P1-P4
    d_p1_p4 = np.linalg.norm(coords[0] - coords[3])

    if d_p1_p4 == 0:
        return 0.0
        
    ear = (d_p2_p6 + d_p3_p5) / (2.0 * d_p1_p4)
    return ear

def estimate_head_pose(landmarks, img_w, img_h, cam_matrix, dist_coeffs):
    # Extract image points
    image_points = []
    for idx in [NOSE_TIP, CHIN, LEFT_EYE_CORNER, RIGHT_EYE_CORNER, LEFT_MOUTH_CORNER, RIGHT_MOUTH_CORNER]:
        lm = landmarks[idx]
        image_points.append((lm.x * img_w, lm.y * img_h))
    
    image_points = np.array(image_points, dtype=np.float64)

    # Solve PnP
    success, rotation_vector, translation_vector = cv2.solvePnP(
        GENERIC_3D_MODEL_POINTS,
        image_points,
        cam_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    
    # Get rotation matrix
    rmat, _ = cv2.Rodrigues(rotation_vector)
    
    # Calculate Euler angles
    # sy = sqrt(R00 * R00 +  R10 * R10)
    sy = np.sqrt(rmat[0, 0] * rmat[0, 0] + rmat[1, 0] * rmat[1, 0])
    
    singular = sy < 1e-6
    if not singular:
        x = np.arctan2(rmat[2, 1], rmat[2, 2])
        y = np.arctan2(-rmat[2, 0], sy)
        z = np.arctan2(rmat[1, 0], rmat[0, 0])
    else:
        x = np.arctan2(-rmat[1, 2], rmat[1, 1])
        y = np.arctan2(-rmat[2, 0], sy)
        z = 0

    return np.degrees(x), np.degrees(y), np.degrees(z) # Pitch, Yaw, Roll

def check_liveness(video_path: str):
    # Initialize Landmarker if not valid
    global face_landmarker
    with _mesh_lock:
        if face_landmarker is None:
             get_face_landmarker()
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"blink_score": 0.0, "head_pose_score": 0.0, "is_liveness_fail": True, "error": "Could not open video"}

    ears = []
    pitch_vals = []
    yaw_vals = []
    roll_vals = []

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            h, w, c = frame.shape
            
            # Camera matrix approximation
            focal_length = w
            center = (w / 2, h / 2)
            cam_matrix = np.array(
                [[focal_length, 0, center[0]],
                 [0, focal_length, center[1]],
                 [0, 0, 1]], dtype="double"
            )
            dist_coeffs = np.zeros((4, 1))

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            # Using the lock for thread safety with the shared global resource
            with _mesh_lock:
                # Detect using Tasks API
                results = face_landmarker.detect(mp_image)

            if results.face_landmarks:
                # face_landmarks is a list of lists of NormalizedLandmark
                for face_landmarks in results.face_landmarks:
                    # face_landmarks is a list of NormalizedLandmark objects which have x, y, z
                    lm_list = face_landmarks
                    
                    # 1. Blink Detection (EAR)
                    left_ear = calculate_ear(lm_list, LEFT_EYE, w, h)
                    right_ear = calculate_ear(lm_list, RIGHT_EYE, w, h)
                    avg_ear = (left_ear + right_ear) / 2.0
                    ears.append(avg_ear)

                    # 2. Head Pose
                    pitch, yaw, roll = estimate_head_pose(lm_list, w, h, cam_matrix, dist_coeffs)
                    pitch_vals.append(pitch)
                    yaw_vals.append(yaw)
                    roll_vals.append(roll)
            else:
                # No face detected in this frame
                pass
                
    finally:
        cap.release()

    # === ANALYSIS ===
    if not ears or not pitch_vals:
         return {"blink_score": 0.0, "head_pose_score": 0.0, "is_liveness_fail": True, "reason": "No face detected"}

    # Blink Logic
    ear_std = np.std(ears)
    ear_mean = np.mean(ears)
    
    blink_score = ear_std 

    # Pose Logic
    pitch_std = np.std(pitch_vals)
    yaw_std = np.std(yaw_vals)
    roll_std = np.std(roll_vals)
    
    head_pose_score = (pitch_std + yaw_std + roll_std) / 3.0

    # Verdict
    is_liveness_fail = False
    
    # 1. Blink Failure Condition: Eyes never blink (very low variance) OR eyes always closed
    if ear_std < 0.002 or ear_mean < 0.15: 
        is_liveness_fail = True
    
    # 2. Pose Failure Condition: Extremely static head (typical of cheap deepfakes or images)
    if head_pose_score < 0.2:
        is_liveness_fail = True

    return {
        "blink_score": float(blink_score),
        "head_pose_score": float(head_pose_score),
        "is_liveness_fail": is_liveness_fail
    }
