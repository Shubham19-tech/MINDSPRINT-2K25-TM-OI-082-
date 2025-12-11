"""
proctor_prototype.py

Prototype real-time video detection for:
- multi-person face detection
- blink counting (per tracked face)
- head pose movement (yaw/pitch/roll -> look-away flag)
- smile detection (simple lip-corner distance heuristic)
- simple mobile detection heuristic: hand close-to-face (proxy for phone)
- event logging and saving evidence frames

Dependencies:
    pip install opencv-python mediapipe numpy

Notes:
- This is a prototype for research/pilot. For production, replace heuristics
  with robust object detectors, identity tracking, anti-spoofing, and privacy compliance.
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import os
import csv
from collections import deque

# ---------- CONFIG ----------
CAMERA_ID = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

BLINK_EAR_THRESH = 0.21        # eye aspect ratio threshold (tunable)
BLINK_CONSEC_FRAMES = 3        # how many consecutive frames EAR < thresh to count a blink
LOOK_AWAY_YAW_THRESH = 25.0    # degrees, yaw beyond -> look-away
LOOK_AWAY_PITCH_THRESH = 20.0  # degrees, pitch beyond -> looking down/up
SMILE_RATIO_THRESH = 0.45      # ratio for smile detection (tunable)
HAND_FACE_DIST_THRESH = 0.15   # normalized distance (hand wrist to nose) threshold for mobile heuristic
MAX_TRACK_DISAPPEAR = 40       # frames to keep disappeared tracks

OUTPUT_DIR = "proctor_evidence"
LOG_FILE = os.path.join(OUTPUT_DIR, "events_log.csv")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------- Mediapipe init ----------
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
                                  max_num_faces=4,
                                  refine_landmarks=True,
                                  min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=2,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)

# ---------- Utilities ----------
def norm_point(p, w, h):
    return int(p.x * w), int(p.y * h)

def euclid(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

# Eye landmark indices for MediaPipe face mesh (refined eye landmarks)
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]   # [left corner, upper inner, upper outer, right corner, lower outer, lower inner]
RIGHT_EYE_IDX = [263, 387, 385, 362, 380, 373]

# Smile detection using lip corners and vertical mouth gap
MOUTH_LEFT = 61
MOUTH_RIGHT = 291
UPPER_LIP = 13
LOWER_LIP = 14

# Face 2D-3D points for head pose (generic model)
# 3D model points (approximate)
MODEL_POINTS_3D = np.array([
    (0.0, 0.0, 0.0),             # Nose tip
    (0.0, -330.0, -65.0),        # Chin
    (-225.0, 170.0, -135.0),     # Left eye left corner
    (225.0, 170.0, -135.0),      # Right eye right corner
    (-150.0, -150.0, -125.0),    # Left Mouth corner
    (150.0, -150.0, -125.0)      # Right mouth corner
], dtype=np.float64)

# ---------- Simple tracker for faces ----------
class TrackedFace:
    _next_id = 0
    def __init__(self, centroid, bbox):
        self.id = TrackedFace._next_id
        TrackedFace._next_id += 1
        self.centroid = centroid
        self.bbox = bbox
        self.disappeared = 0
        self.blink_count = 0
        self.consec_closed = 0
        self.last_smile = False
        self.last_head_pose = (0.0, 0.0, 0.0)
        self.events = deque(maxlen=50)

    def mark_disappeared(self):
        self.disappeared += 1

    def mark_seen(self, centroid, bbox):
        self.centroid = centroid
        self.bbox = bbox
        self.disappeared = 0

# ---------- Tracker management ----------
tracked_faces = []

def match_faces(detected_centroids, detected_bboxes, max_distance=80):
    """
    Greedy nearest neighbor assignment from detected_centroids to tracked_faces.
    Creates new tracks for unmatched detections and marks disappeared for missing ones.
    """
    global tracked_faces
    assigned = set()
    # if no current tracks, create tracks
    if len(tracked_faces) == 0:
        for c, b in zip(detected_centroids, detected_bboxes):
            tracked_faces.append(TrackedFace(c, b))
        return

    # compute distance matrix
    distances = []
    for i, t in enumerate(tracked_faces):
        distances.append([euclid(t.centroid, c) for c in detected_centroids])
    distances = np.array(distances) if len(distances)>0 else np.zeros((0,len(detected_centroids)))

    # greedy matching
    if distances.size > 0:
        while True:
            i,j = np.unravel_index(np.argmin(distances), distances.shape)
            if distances[i,j] > max_distance:
                break
            if distances[i,j] == np.inf:
                break
            # assign
            tracked_faces[i].mark_seen(detected_centroids[j], detected_bboxes[j])
            assigned.add(j)
            # invalidate row and column
            distances[i,:] = np.inf
            distances[:,j] = np.inf
            if np.all(distances == np.inf):
                break

    # unmatched detections -> new tracks
    for idx, (c,b) in enumerate(zip(detected_centroids, detected_bboxes)):
        if idx not in assigned:
            tracked_faces.append(TrackedFace(c,b))

    # increment disappeared for unassigned tracks
    for t in tracked_faces:
        # if this track wasn't updated in this frame, it will still have the old centroid.
        # We consider it updated if disappeared==0 (we reset disappeared in mark_seen)
        if t.disappeared == 0 and t.centroid not in detected_centroids:
            # already seen this frame? skip
            pass
    # Mark disappeared for tracks not assigned (we can approximate by checking if they were matched above)
    current_centroids = detected_centroids
    for t in tracked_faces:
        if all(euclid(t.centroid, c) > 1e-6 for c in current_centroids):
            t.mark_disappeared()

    # remove long-disappeared
    tracked_faces = [t for t in tracked_faces if t.disappeared <= MAX_TRACK_DISAPPEAR]

# ---------- EAR and smile helpers ----------
def eye_aspect_ratio(landmarks, eye_idx, image_w, image_h):
    pts = []
    for idx in eye_idx:
        lm = landmarks[idx]
        pts.append((lm.x * image_w, lm.y * image_h))
    # EAR formula using 6 points: vertical distances / horizontal distance
    A = euclid(pts[1], pts[5])
    B = euclid(pts[2], pts[4])
    C = euclid(pts[0], pts[3])
    if C == 0:
        return 0.0
    ear = (A + B) / (2.0 * C)
    return ear

def smile_ratio(landmarks, image_w, image_h):
    left = landmarks[MOUTH_LEFT]
    right = landmarks[MOUTH_RIGHT]
    upper = landmarks[UPPER_LIP]
    lower = landmarks[LOWER_LIP]
    horiz = euclid((left.x*image_w, left.y*image_h), (right.x*image_w, right.y*image_h))
    vert = euclid((upper.x*image_w, upper.y*image_h), (lower.x*image_w, lower.y*image_h))
    if horiz == 0:
        return 0.0
    return vert / horiz

# ---------- Head pose ----------
def estimate_head_pose(landmarks, image_w, image_h, camera_matrix=None, dist_coeffs=None):
    # select 2D image points from mediapipe landmarks
    try:
        image_points = np.array([
            (landmarks[1].x * image_w, landmarks[1].y * image_h),    # nose tip (mp index 1)
            (landmarks[199].x * image_w, landmarks[199].y * image_h),# chin approx (use index 199)
            (landmarks[33].x * image_w, landmarks[33].y * image_h),  # left eye left corner
            (landmarks[263].x * image_w, landmarks[263].y * image_h),# right eye right corner
            (landmarks[61].x * image_w, landmarks[61].y * image_h),  # left mouth corner
            (landmarks[291].x * image_w, landmarks[291].y * image_h) # right mouth corner
        ], dtype=np.float64)
    except Exception as e:
        return (0.0, 0.0, 0.0)

    if camera_matrix is None:
        focal_length = image_w
        center = (image_w/2, image_h/2)
        camera_matrix = np.array([[focal_length, 0, center[0]],
                                  [0, focal_length, center[1]],
                                  [0, 0, 1]], dtype=np.float64)
    if dist_coeffs is None:
        dist_coeffs = np.zeros((4,1))  # assume no lens distortion

    success, rotation_vector, translation_vector = cv2.solvePnP(MODEL_POINTS_3D, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    if not success:
        return (0.0, 0.0, 0.0)
    rmat, _ = cv2.Rodrigues(rotation_vector)
    pose_mat = cv2.hconcat((rmat, translation_vector))
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)
    pitch, yaw, roll = [float(x) for x in euler_angles]
    return (yaw, pitch, roll)

# ---------- Logging ----------
def log_event(face_id, event_type, details, frame, timestamp=None):
    if timestamp is None:
        timestamp = time.time()
    timestr = time.strftime("%Y%m%d_%H%M%S", time.localtime(timestamp))
    fname = f"{OUTPUT_DIR}/face{face_id}_{event_type}_{timestr}.jpg"
    cv2.imwrite(fname, frame)
    # append to CSV
    header_needed = not os.path.exists(LOG_FILE)
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if header_needed:
            writer.writerow(["timestamp", "face_id", "event", "details", "evidence"])
        writer.writerow([timestamp, face_id, event_type, details, fname])

# ---------- Main loop ----------
def main():
    cap = cv2.VideoCapture(CAMERA_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    # camera intrinsics for head pose
    image_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    image_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    focal_length = image_w
    cam_matrix = np.array([[focal_length, 0, image_w/2],
                           [0, focal_length, image_h/2],
                           [0, 0, 1]], dtype=np.float64)

    global tracked_faces
    tracked_faces = []

    print("[INFO] Starting video. Press 'q' to quit.")
    fps_t0 = time.time()
    frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Camera frame not read.")
            break
        frames += 1
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Face detection (bounding boxes) for multi-person detection / coarse centroids
        det_results = face_detection.process(rgb)
        detected_centroids = []
        detected_bboxes = []

        if det_results.detections:
            for det in det_results.detections:
                # bounding box relative
                bbox = det.location_data.relative_bounding_box
                x1 = int(bbox.xmin * image_w)
                y1 = int(bbox.ymin * image_h)
                w = int(bbox.width * image_w)
                h = int(bbox.height * image_h)
                x2, y2 = x1 + w, y1 + h
                cx = x1 + w // 2
                cy = y1 + h // 2
                detected_centroids.append((cx, cy))
                detected_bboxes.append((x1, y1, x2, y2))

        # Update simple face tracker
        match_faces(detected_centroids, detected_bboxes)

        # Face mesh for landmarks (gives per-face landmarks)
        mesh_results = face_mesh.process(rgb)

        # Hands for mobile heuristic
        hands_results = hands.process(rgb)
        hands_landmarks = []
        if hands_results.multi_hand_landmarks:
            for hand_landmarks in hands_results.multi_hand_landmarks:
                hands_landmarks.append(hand_landmarks)

        # For each tracked face, try to match a mesh face
        face_id_to_mesh = {}  # mapping tracked id -> mediapipe face_landmarks
        if mesh_results.multi_face_landmarks:
            # compute centroids of mesh faces and match to tracker by nearest centroid
            mesh_centroids = []
            for mesh in mesh_results.multi_face_landmarks:
                xs = [lm.x for lm in mesh.landmark]
                ys = [lm.y for lm in mesh.landmark]
                cx = int(np.mean(xs) * image_w)
                cy = int(np.mean(ys) * image_h)
                mesh_centroids.append((cx, cy))
            # match each mesh to nearest tracked face id
            for i, mesh_cent in enumerate(mesh_centroids):
                best_id = None
                best_dist = 1e9
                for t in tracked_faces:
                    d = euclid(t.centroid, mesh_cent)
                    if d < best_dist:
                        best_dist = d
                        best_id = t.id
                if best_id is not None:
                    face_id_to_mesh[best_id] = mesh_results.multi_face_landmarks[i]

        # Process each tracked face
        for t in tracked_faces:
            # draw bbox/ID if available
            (x1, y1, x2, y2) = t.bbox if t.bbox is not None else (0,0,0,0)
            cv2.rectangle(frame, (x1,y1), (x2,y2), (10,200,50), 2)
            cv2.putText(frame, f"ID:{t.id}", (x1, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

            if t.id in face_id_to_mesh:
                mesh = face_id_to_mesh[t.id]
                landmarks = mesh.landmark

                # draw landmarks (optional)
                mp_drawing.draw_landmarks(frame, mesh, mp_face_mesh.FACEMESH_TESSELATION,
                                          mp_drawing.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=1),
                                          mp_drawing.DrawingSpec(color=(0,128,255), thickness=1))

                # EAR blink detection (take average of two eyes)
                left_ear = eye_aspect_ratio(landmarks, LEFT_EYE_IDX, image_w, image_h)
                right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE_IDX, image_w, image_h)
                ear = (left_ear + right_ear) / 2.0

                # blink logic
                if ear < BLINK_EAR_THRESH:
                    t.consec_closed += 1
                else:
                    if t.consec_closed >= BLINK_CONSEC_FRAMES:
                        t.blink_count += 1
                        t.events.append(("blink", time.time()))
                        # save evidence
                        log_event(t.id, "blink", f"blink_count={t.blink_count}", frame)
                    t.consec_closed = 0

                # smile detection
                sratio = smile_ratio(landmarks, image_w, image_h)
                is_smile = sratio > SMILE_RATIO_THRESH
                if is_smile and not t.last_smile:
                    t.events.append(("smile", time.time()))
                    log_event(t.id, "smile", f"smile_ratio={sratio:.3f}", frame)
                t.last_smile = is_smile

                # head pose
                yaw, pitch, roll = estimate_head_pose(landmarks, image_w, image_h, camera_matrix=cam_matrix)
                t.last_head_pose = (yaw, pitch, roll)
                # flag look-away if yaw or pitch beyond threshold
                if abs(yaw) > LOOK_AWAY_YAW_THRESH or abs(pitch) > LOOK_AWAY_PITCH_THRESH:
                    t.events.append(("look_away", time.time()))
                    log_event(t.id, "look_away", f"yaw={yaw:.1f},pitch={pitch:.1f}", frame)

                # overlay some text
                cv2.putText(frame, f"Blink:{t.blink_count}", (x1, y2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                cv2.putText(frame, f"Smile:{'Y' if is_smile else 'N'}", (x1, y2+45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                cv2.putText(frame, f"Yaw:{yaw:.1f}", (x1, y2+70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

                # mobile detection heuristic: any hand wrist close to nose -> possible phone
                nose = (landmarks[1].x * image_w, landmarks[1].y * image_h)
                phone_sus = False
                if hands_landmarks:
                    for hland in hands_landmarks:
                        wrist = hland.landmark[0]  # wrist
                        wrist_xy = (wrist.x * image_w, wrist.y * image_h)
                        # normalized by image diagonal
                        norm_dist = euclid(wrist_xy, nose) / np.sqrt(image_w**2 + image_h**2)
                        if norm_dist < HAND_FACE_DIST_THRESH:
                            phone_sus = True
                            t.events.append(("hand_near_face", time.time()))
                            log_event(t.id, "hand_near_face", f"norm_dist={norm_dist:.3f}", frame)
                            break
                if phone_sus:
                    cv2.putText(frame, "MOBILE_SUSPECT", (x1, y2+95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

            else:
                # no mesh associated -> just draw ID
                pass

        # Global multi-person detection flag
        multi_count = len(tracked_faces)
        cv2.putText(frame, f"People: {multi_count}", (20,30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,255), 2)
        if multi_count > 1:
            # log multi-person if seen this frame
            # Save evidence once per 40 frames approx (to avoid rapid duplicates)
            if frames % 40 == 0:
                log_event(-1, "multi_person", f"count={multi_count}", frame)
            cv2.putText(frame, "MULTI_PERSON_DETECTED", (20,60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

        # draw hands (optional)
        if hands_results.multi_hand_landmarks:
            for hand_landmarks in hands_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # show FPS
        if frames % 15 == 0:
            fps = 15 / (time.time() - fps_t0)
            fps_t0 = time.time()
            cv2.putText(frame, f"FPS:{fps:.1f}", (image_w-140, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

        cv2.imshow("Proctor Prototype", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('s'):
            # save snapshot
            ts = time.strftime("%Y%m%d_%H%M%S")
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"snapshot_{ts}.jpg"), frame)
            print("[INFO] Snapshot saved.")

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Finished.")

if __name__ == "__main__":
    main()
