# Live Gestik & Mimik Scanner

## Beschreibung
Das Programm wurde zu einem Live-Scanner umgebaut, der kontinuierlich die Webcam auswertet und Gesichter sowie deren Emotionen in Echtzeit erkennt.

## Voraussetzungen
```bash
pip install torch torchvision
pip install transformers
pip install opencv-python
pip install Pillow
pip install numpy
```

## Verwendung
Starte das Programm mit:
```bash
python blazeface.py
```

Das Programm:
- Öffnet automatisch deine Webcam
- Zeigt ein Live-Video-Fenster an
- Erkennt Gesichter in Echtzeit (grüne Bounding Boxes)
- Zeigt erkannte Emotionen über jedem Gesicht an (rot)
- Markiert Gesichts-Keypoints (gelb)
- Zeigt FPS und Anzahl erkannter Gesichter an

## Bedienung
- **'q' drücken**: Beendet den Scanner
- Das Programm läuft kontinuierlich bis zum Beenden

## Features
- ✓ Live Webcam-Feed
- ✓ Echtzeit-Gesichtserkennung
- ✓ Echtzeit-Emotions-Erkennung
- ✓ FPS-Anzeige
- ✓ Gesichts-Keypoint-Visualisierung
- ✓ Mehrere Gesichter gleichzeitig

## Erkannte Emotionen
Das Modell erkennt verschiedene Emotionen wie:
- Happy / Fröhlich
- Sad / Traurig
- Angry / Wütend
- Neutral
- und weitere
