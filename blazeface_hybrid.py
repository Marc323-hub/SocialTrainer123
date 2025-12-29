"""
Hybrid Live-Gestik/Mimik-Scanner
- Funktioniert MIT und OHNE MediaPipe
- Automatischer Fallback auf BlazeFace
"""

import cv2
import time
import numpy as np
from PIL import Image, ImageDraw
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch
import torch.nn as nn
import torch.nn.functional as F

# Versuche MediaPipe zu importieren UND zu testen
try:
    import mediapipe as mp
    # Teste ob solutions verfügbar ist (Python 3.14 Kompatibilität)
    _ = mp.solutions.face_mesh
    MEDIAPIPE_AVAILABLE = True
    print("✓ MediaPipe gefunden - verwende 468 Landmarks")
except (ImportError, AttributeError) as e:
    MEDIAPIPE_AVAILABLE = False
    print("✗ MediaPipe nicht verfügbar - verwende BlazeFace mit 6 Keypoints")
    if 'mediapipe' in str(e):
        print("  (Hinweis: Python 3.14 wird von MediaPipe noch nicht vollständig unterstützt)")

# === EINSTELLUNGEN ===
EMO_MODEL_ID = "trpakov/vit-face-expression"
WEIGHTS_PATH = "blazeface.pth"
ANCHORS_PATH = "anchors.npy"
# ====================

# BlazeFace Klassen (nur wenn MediaPipe nicht verfügbar)
if not MEDIAPIPE_AVAILABLE:
    class BlazeBlock(nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
            super(BlazeBlock, self).__init__()
            self.stride = stride
            self.channel_pad = out_channels - in_channels

            if stride == 2:
                self.max_pool = nn.MaxPool2d(kernel_size=stride, stride=stride)
                padding = 0
            else:
                padding = (kernel_size - 1) // 2

            self.convs = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=in_channels,
                          kernel_size=kernel_size, stride=stride, padding=padding,
                          groups=in_channels, bias=True),
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                          kernel_size=1, stride=1, padding=0, bias=True),
            )
            self.act = nn.ReLU(inplace=True)

        def forward(self, x):
            if self.stride == 2:
                h = F.pad(x, (0, 2, 0, 2), "constant", 0)
                x = self.max_pool(x)
            else:
                h = x

            if self.channel_pad > 0:
                x = F.pad(x, (0, 0, 0, 0, 0, self.channel_pad), "constant", 0)

            return self.act(self.convs(h) + x)

    class BlazeFace(nn.Module):
        def __init__(self):
            super(BlazeFace, self).__init__()
            self.num_classes = 1
            self.num_anchors = 896
            self.num_coords = 16
            self.score_clipping_thresh = 100.0
            self.x_scale = 128.0
            self.y_scale = 128.0
            self.h_scale = 128.0
            self.w_scale = 128.0
            self.min_score_thresh = 0.75
            self.min_suppression_threshold = 0.3

            self.backbone1 = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=24, kernel_size=5, stride=2, padding=0, bias=True),
                nn.ReLU(inplace=True),
                BlazeBlock(24, 24),
                BlazeBlock(24, 28),
                BlazeBlock(28, 32, stride=2),
                BlazeBlock(32, 36),
                BlazeBlock(36, 42),
                BlazeBlock(42, 48, stride=2),
                BlazeBlock(48, 56),
                BlazeBlock(56, 64),
                BlazeBlock(64, 72),
                BlazeBlock(72, 80),
                BlazeBlock(80, 88),
            )

            self.backbone2 = nn.Sequential(
                BlazeBlock(88, 96, stride=2),
                BlazeBlock(96, 96),
                BlazeBlock(96, 96),
                BlazeBlock(96, 96),
                BlazeBlock(96, 96),
            )

            self.classifier_8 = nn.Conv2d(88, 2, 1, bias=True)
            self.classifier_16 = nn.Conv2d(96, 6, 1, bias=True)
            self.regressor_8 = nn.Conv2d(88, 32, 1, bias=True)
            self.regressor_16 = nn.Conv2d(96, 96, 1, bias=True)

        def forward(self, x):
            x = F.pad(x, (1, 2, 1, 2), "constant", 0)
            b = x.shape[0]

            x = self.backbone1(x)
            h = self.backbone2(x)

            c1 = self.classifier_8(x).permute(0, 2, 3, 1).reshape(b, -1, 1)
            c2 = self.classifier_16(h).permute(0, 2, 3, 1).reshape(b, -1, 1)
            c = torch.cat((c1, c2), dim=1)

            r1 = self.regressor_8(x).permute(0, 2, 3, 1).reshape(b, -1, 16)
            r2 = self.regressor_16(h).permute(0, 2, 3, 1).reshape(b, -1, 16)
            r = torch.cat((r1, r2), dim=1)

            return [r, c]

        def _device(self):
            return self.classifier_8.weight.device

        def load_weights(self, path):
            state = torch.load(path, map_location=self._device(), weights_only=True)
            self.load_state_dict(state)
            self.eval()

        def load_anchors(self, path):
            self.anchors = torch.tensor(np.load(path), dtype=torch.float32, device=self._device())

        def _preprocess(self, x):
            return x.float() / 127.5 - 1.0

        def predict_on_image(self, img):
            if isinstance(img, np.ndarray):
                img = torch.from_numpy(img).permute((2, 0, 1))
            return self.predict_on_batch(img.unsqueeze(0))[0]

        def predict_on_batch(self, x):
            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x).permute((0, 3, 1, 2))

            x = x.to(self._device())
            x = self._preprocess(x)

            with torch.no_grad():
                out = self.__call__(x)

            detections = self._tensors_to_detections(out[0], out[1], self.anchors)

            filtered_detections = []
            for i in range(len(detections)):
                faces = self._weighted_non_max_suppression(detections[i])
                faces = torch.stack(faces) if len(faces) > 0 else torch.zeros((0, 17))
                filtered_detections.append(faces)

            return filtered_detections

        def _tensors_to_detections(self, raw_box_tensor, raw_score_tensor, anchors):
            detection_boxes = self._decode_boxes(raw_box_tensor, anchors)
            thresh = self.score_clipping_thresh
            raw_score_tensor = raw_score_tensor.clamp(-thresh, thresh)
            detection_scores = raw_score_tensor.sigmoid().squeeze(dim=-1)
            mask = detection_scores >= self.min_score_thresh

            output_detections = []
            for i in range(raw_box_tensor.shape[0]):
                boxes = detection_boxes[i, mask[i]]
                scores = detection_scores[i, mask[i]].unsqueeze(dim=-1)
                output_detections.append(torch.cat((boxes, scores), dim=-1))

            return output_detections

        def _decode_boxes(self, raw_boxes, anchors):
            boxes = torch.zeros_like(raw_boxes)
            x_center = raw_boxes[..., 0] / self.x_scale * anchors[:, 2] + anchors[:, 0]
            y_center = raw_boxes[..., 1] / self.y_scale * anchors[:, 3] + anchors[:, 1]
            w = raw_boxes[..., 2] / self.w_scale * anchors[:, 2]
            h = raw_boxes[..., 3] / self.h_scale * anchors[:, 3]

            boxes[..., 0] = y_center - h / 2.
            boxes[..., 1] = x_center - w / 2.
            boxes[..., 2] = y_center + h / 2.
            boxes[..., 3] = x_center + w / 2.

            for k in range(6):
                offset = 4 + k*2
                keypoint_x = raw_boxes[..., offset] / self.x_scale * anchors[:, 2] + anchors[:, 0]
                keypoint_y = raw_boxes[..., offset + 1] / self.y_scale * anchors[:, 3] + anchors[:, 1]
                boxes[..., offset] = keypoint_x
                boxes[..., offset + 1] = keypoint_y

            return boxes

        def _weighted_non_max_suppression(self, detections):
            if len(detections) == 0: return []

            output_detections = []
            remaining = torch.argsort(detections[:, 16], descending=True)

            while len(remaining) > 0:
                detection = detections[remaining[0]]
                first_box = detection[:4]
                other_boxes = detections[remaining, :4]
                ious = self._overlap_similarity(first_box, other_boxes)

                mask = ious > self.min_suppression_threshold
                overlapping = remaining[mask]
                remaining = remaining[~mask]

                weighted_detection = detection.clone()
                if len(overlapping) > 1:
                    coordinates = detections[overlapping, :16]
                    scores = detections[overlapping, 16:17]
                    total_score = scores.sum()
                    weighted = (coordinates * scores).sum(dim=0) / total_score
                    weighted_detection[:16] = weighted
                    weighted_detection[16] = total_score / len(overlapping)

                output_detections.append(weighted_detection)

            return output_detections

        def _overlap_similarity(self, box, other_boxes):
            box_a = box.unsqueeze(0)
            box_b = other_boxes

            A = box_a.size(0)
            B = box_b.size(0)
            max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                               box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
            min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                               box_b[:, :2].unsqueeze(0).expand(A, B, 2))
            inter = torch.clamp((max_xy - min_xy), min=0)
            inter = inter[:, :, 0] * inter[:, :, 1]

            area_a = ((box_a[:, 2]-box_a[:, 0]) * (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)
            area_b = ((box_b[:, 2]-box_b[:, 0]) * (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter)
            union = area_a + area_b - inter

            return (inter / union).squeeze(0)

# Emotions-Modell laden
print("Lade Emotions-Erkennungsmodell...")
emo_processor = AutoImageProcessor.from_pretrained(EMO_MODEL_ID)
emo_model = AutoModelForImageClassification.from_pretrained(EMO_MODEL_ID)
emo_model.eval()

def predict_expression(face_pil):
    inputs = emo_processor(images=face_pil, return_tensors="pt")
    with torch.no_grad():
        logits = emo_model(**inputs).logits
        probs = logits.softmax(dim=-1)[0]

    idx = int(probs.argmax().item())
    label = emo_model.config.id2label[idx]
    conf = float(probs[idx].item())

    emotion_map = {
        'angry': 'Wütend', 'disgust': 'Ekel', 'fear': 'Angst',
        'happy': 'Fröhlich', 'sad': 'Traurig', 'surprise': 'Überrascht',
        'neutral': 'Neutral'
    }
    return emotion_map.get(label, label), conf

# Initialisierung basierend auf verfügbarer Technologie
if MEDIAPIPE_AVAILABLE:
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=5,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    scanner_mode = "MediaPipe (468 Landmarks)"
else:
    print("Lade BlazeFace-Modell...")
    face_detector = BlazeFace()
    face_detector.load_weights(WEIGHTS_PATH)
    face_detector.load_anchors(ANCHORS_PATH)
    scanner_mode = "BlazeFace (6 Keypoints)"

print("\nÖffne Webcam...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("FEHLER: Webcam konnte nicht geöffnet werden!")
    exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("\n" + "="*70)
print(f"HYBRID LIVE GESTIK & MIMIK SCANNER - {scanner_mode}")
print("="*70)
print("Steuerung: 'q' = Beenden")
print("="*70 + "\n")

frame_count = 0
fps_start_time = time.time()
fps = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Fehler beim Lesen des Webcam-Frames")
        break

    frame_count += 1
    if frame_count % 30 == 0:
        fps = 30 / (time.time() - fps_start_time)
        fps_start_time = time.time()

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    H, W = frame.shape[:2]
    num_faces = 0

    if MEDIAPIPE_AVAILABLE:
        # MediaPipe Verarbeitung
        results = face_mesh.process(frame_rgb)

        if results.multi_face_landmarks:
            num_faces = len(results.multi_face_landmarks)

            for face_landmarks in results.multi_face_landmarks:
                x_coords = [lm.x for lm in face_landmarks.landmark]
                y_coords = [lm.y for lm in face_landmarks.landmark]

                x_min = int(min(x_coords) * W)
                y_min = int(min(y_coords) * H)
                x_max = int(max(x_coords) * W)
                y_max = int(max(y_coords) * H)

                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                # Wichtige Landmarks zeichnen
                for idx in [33, 133, 263, 362, 1, 61, 291]:  # Augen, Nase, Mund
                    lm = face_landmarks.landmark[idx]
                    x, y = int(lm.x * W), int(lm.y * H)
                    cv2.circle(frame, (x, y), 2, (255, 255, 0), -1)

                # Emotion erkennen
                pad = 20
                x0 = max(0, x_min - pad)
                y0 = max(0, y_min - pad)
                x1 = min(W, x_max + pad)
                y1 = min(H, y_max + pad)

                if x1 > x0 and y1 > y0:
                    face_crop = Image.fromarray(frame_rgb[y0:y1, x0:x1])
                    emotion_label, emotion_conf = predict_expression(face_crop)

                    cv2.putText(frame, f"{emotion_label} ({emotion_conf:.2f})",
                               (x_min, max(0, y_min - 10)),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        # BlazeFace Verarbeitung
        frame_128 = cv2.resize(frame_rgb, (128, 128))
        detections = face_detector.predict_on_image(frame_128)
        num_faces = detections.shape[0]

        if num_faces > 0:
            for det in detections:
                ymin, xmin, ymax, xmax = det[:4].tolist()
                score = float(det[16])

                x0 = int(xmin * W)
                y0 = int(ymin * H)
                x1 = int(xmax * W)
                y1 = int(ymax * H)

                cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 2)

                # 6 Keypoints
                for k in range(6):
                    px = int(float(det[4 + 2*k]) * W)
                    py = int(float(det[4 + 2*k + 1]) * H)
                    cv2.circle(frame, (px, py), 3, (255, 255, 0), -1)

                # Emotion
                pad = 20
                x0_crop = max(0, x0 - pad)
                y0_crop = max(0, y0 - pad)
                x1_crop = min(W, x1 + pad)
                y1_crop = min(H, y1 + pad)

                if x1_crop > x0_crop and y1_crop > y0_crop:
                    face_crop = Image.fromarray(frame_rgb[y0_crop:y1_crop, x0_crop:x1_crop])
                    emotion_label, emotion_conf = predict_expression(face_crop)

                    cv2.putText(frame, f"{emotion_label} ({emotion_conf:.2f})",
                               (x0, max(0, y0 - 10)),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Status
    status_text = f"FPS: {fps:.1f} | Gesichter: {num_faces} | Modus: {scanner_mode}"
    cv2.putText(frame, status_text, (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.putText(frame, "q=Beenden", (10, H - 10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow('Hybrid Gestik & Mimik Scanner', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("\nBeende Scanner...")
        break

cap.release()
cv2.destroyAllWindows()
if MEDIAPIPE_AVAILABLE:
    face_mesh.close()
print("Scanner beendet.")
