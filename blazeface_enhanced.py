"""
Verbesserter Live-Gestik/Mimik-Scanner mit:
- MediaPipe Face Mesh (468 Landmarks)
- Erweiterte Emotions-Erkennung
- Präzisere Gesichtserkennung
"""

import cv2
import time
import numpy as np
from PIL import Image
import mediapipe as mp
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch

# === EINSTELLUNGEN ===
EMO_MODEL_ID = "trpakov/vit-face-expression"  # Besseres Modell
SHOW_ALL_LANDMARKS = False  # True = alle 468 Punkte, False = nur wichtige Punkte
# ====================

print("Initialisiere MediaPipe Face Mesh...")
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Face Mesh initialisieren
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=5,
    refine_landmarks=True,  # Zusätzliche Iris-Landmarks
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

print("Lade Emotions-Erkennungsmodell...")
emo_processor = AutoImageProcessor.from_pretrained(EMO_MODEL_ID)
emo_model = AutoModelForImageClassification.from_pretrained(EMO_MODEL_ID)
emo_model.eval()

def predict_expression(face_pil):
    """Erkennt die Emotion eines Gesichts-Crops."""
    inputs = emo_processor(images=face_pil, return_tensors="pt")

    with torch.no_grad():
        logits = emo_model(**inputs).logits
        probs = logits.softmax(dim=-1)[0]

    idx = int(probs.argmax().item())
    label = emo_model.config.id2label[idx]
    conf = float(probs[idx].item())

    # Deutsche Übersetzung
    emotion_map = {
        'angry': 'Wütend',
        'disgust': 'Ekel',
        'fear': 'Angst',
        'happy': 'Fröhlich',
        'sad': 'Traurig',
        'surprise': 'Überrascht',
        'neutral': 'Neutral'
    }

    label_de = emotion_map.get(label, label)
    return label_de, conf

# Wichtige Gesichtspunkte für visuelle Darstellung
IMPORTANT_LANDMARKS = {
    'left_eye': [33, 133, 160, 159, 158, 157, 173, 246],
    'right_eye': [263, 362, 387, 386, 385, 384, 398, 466],
    'left_eyebrow': [70, 63, 105, 66, 107],
    'right_eyebrow': [336, 296, 334, 293, 300],
    'nose': [1, 2, 98, 327],
    'mouth': [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291],
    'jaw': [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162]
}

print("\nÖffne Webcam...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("FEHLER: Webcam konnte nicht geöffnet werden!")
    exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("\n" + "="*70)
print("VERBESSERTER LIVE GESTIK & MIMIK SCANNER")
print("="*70)
print("Features:")
print("  - MediaPipe Face Mesh (468 Landmarks)")
print("  - Erweiterte Emotions-Erkennung")
print("  - Präzise Gesichtserkennung")
print("="*70)
print("Steuerung:")
print("  'q' - Beenden")
print("  'l' - Landmark-Modus umschalten (alle/wichtige)")
print("="*70 + "\n")

frame_count = 0
fps_start_time = time.time()
fps = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Fehler beim Lesen des Webcam-Frames")
        break

    # FPS berechnen
    frame_count += 1
    if frame_count % 30 == 0:
        fps = 30 / (time.time() - fps_start_time)
        fps_start_time = time.time()

    # Frame für MediaPipe vorbereiten (RGB)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    H, W = frame.shape[:2]

    # Face Mesh Verarbeitung
    results = face_mesh.process(frame_rgb)

    num_faces = 0

    if results.multi_face_landmarks:
        num_faces = len(results.multi_face_landmarks)

        for face_idx, face_landmarks in enumerate(results.multi_face_landmarks):
            # Bounding Box berechnen
            x_coords = [lm.x for lm in face_landmarks.landmark]
            y_coords = [lm.y for lm in face_landmarks.landmark]

            x_min = int(min(x_coords) * W)
            y_min = int(min(y_coords) * H)
            x_max = int(max(x_coords) * W)
            y_max = int(max(y_coords) * H)

            # Bounding Box zeichnen
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            # Landmarks zeichnen
            if SHOW_ALL_LANDMARKS:
                # Alle 468 Landmarks zeichnen
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                )
            else:
                # Nur wichtige Landmarks zeichnen
                for region_name, landmark_ids in IMPORTANT_LANDMARKS.items():
                    for lm_id in landmark_ids:
                        lm = face_landmarks.landmark[lm_id]
                        x = int(lm.x * W)
                        y = int(lm.y * H)

                        # Farbe je nach Region
                        if 'eye' in region_name:
                            color = (255, 255, 0)  # Gelb
                        elif 'eyebrow' in region_name:
                            color = (0, 255, 255)  # Cyan
                        elif 'nose' in region_name:
                            color = (255, 0, 255)  # Magenta
                        elif 'mouth' in region_name:
                            color = (0, 0, 255)    # Rot
                        else:
                            color = (128, 128, 128)  # Grau

                        cv2.circle(frame, (x, y), 2, color, -1)

            # Gesicht für Emotions-Erkennung croppen
            pad = 20
            x0_crop = max(0, x_min - pad)
            y0_crop = max(0, y_min - pad)
            x1_crop = min(W, x_max + pad)
            y1_crop = min(H, y_max + pad)

            if x1_crop > x0_crop and y1_crop > y0_crop:
                face_crop_np = frame_rgb[y0_crop:y1_crop, x0_crop:x1_crop]

                if face_crop_np.size > 0:
                    face_crop_pil = Image.fromarray(face_crop_np)
                    emotion_label, emotion_conf = predict_expression(face_crop_pil)

                    # Emotion anzeigen
                    emotion_text = f"{emotion_label} ({emotion_conf:.2f})"
                    cv2.putText(frame, emotion_text, (x_min, max(0, y_min - 10)),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Anzahl Landmarks anzeigen
            landmark_text = f"Landmarks: 468"
            cv2.putText(frame, landmark_text, (x_min, max(0, y_min - 35)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Status-Informationen
    status_text = f"FPS: {fps:.1f} | Gesichter: {num_faces}"
    cv2.putText(frame, status_text, (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    mode_text = f"Modus: {'Alle Landmarks' if SHOW_ALL_LANDMARKS else 'Wichtige Punkte'}"
    cv2.putText(frame, mode_text, (10, 60),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.putText(frame, "q=Beenden | l=Landmark-Modus", (10, H - 10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Frame anzeigen
    cv2.imshow('Verbesserter Gestik & Mimik Scanner', frame)

    # Tastatur-Eingabe
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("\nBeende Scanner...")
        break
    elif key == ord('l'):
        SHOW_ALL_LANDMARKS = not SHOW_ALL_LANDMARKS
        print(f"Landmark-Modus: {'Alle 468 Landmarks' if SHOW_ALL_LANDMARKS else 'Nur wichtige Punkte'}")

# Aufräumen
cap.release()
cv2.destroyAllWindows()
face_mesh.close()
print("Scanner beendet.")
