import os
import cv2
import mediapipe as mp
import numpy as np

cap = cv2.VideoCapture(0)

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_dstyles = mp.solutions.drawing_styles

my_drawing_spec = mp_drawing.DrawingSpec(color=(194, 97, 190), thickness=1)

with mp_face_mesh.FaceMesh( 
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        results = face_mesh.process(image)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=my_drawing_spec
                )

                # Save landmarks safely
                landmarks = np.array([[lm.x, lm.y, lm.z] for lm in face_landmarks.landmark])
                tmp_file = 'landmarks_tmp.npy'
                final_file = 'landmarks.npy'
                np.save(tmp_file, landmarks)
                os.replace(tmp_file, final_file)  # atomic rename

                print(f"[Saved] {len(landmarks)} landmarks")

        cv2.imshow("Face Mesh", cv2.flip(image, 1))

        if cv2.waitKey(100) == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
