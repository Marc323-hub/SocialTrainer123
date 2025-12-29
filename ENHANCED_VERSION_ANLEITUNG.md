# Verbesserte Version - Live Gestik & Mimik Scanner

## Was ist neu?

### **468 Gesichtspunkte statt 6**
- Verwendet MediaPipe Face Mesh
- Erkennt:
  - Augen (linkes + rechtes Auge)
  - Augenbrauen
  - Nase
  - Mund (detaillierte Kontur)
  - Kieferlinie
  - Iris (zusätzlich)

### **Bessere Emotions-Erkennung**
- Präziseres Modell
- Gleiche Emotionen, aber genauer erkannt

### **Zwei Anzeigemodi**
- **Wichtige Punkte**: Zeigt die wichtigsten Landmarks (übersichtlich)
- **Alle Landmarks**: Zeigt alle 468 Punkte (sehr detailliert)

## Installation

```bash
# MediaPipe installieren
pip install mediapipe>=0.10.0

# Oder alle Abhängigkeiten neu installieren
pip install -r requirements.txt
```

## Verwendung

```bash
python blazeface_enhanced.py
```

## Steuerung

- **'q'** - Scanner beenden
- **'l'** - Zwischen Landmark-Modi umschalten (wichtige Punkte ↔ alle 468 Punkte)

## Was du siehst

### Farbcodierung der Landmarks:
- **Gelb** - Augen
- **Cyan** - Augenbrauen
- **Magenta** - Nase
- **Rot** - Mund
- **Grau** - Kieferlinie

### Anzeige:
- Grüne Bounding Box um Gesichter
- Emotion in Rot über dem Gesicht
- FPS und Anzahl erkannter Gesichter oben links
- Aktueller Modus (Wichtige Punkte / Alle Landmarks)

## Erkannte Emotionen

Das Modell erkennt:
1. **Wütend** (angry)
2. **Ekel** (disgust)
3. **Angst** (fear)
4. **Fröhlich** (happy)
5. **Traurig** (sad)
6. **Überrascht** (surprise)
7. **Neutral** (neutral)

## Für noch mehr Emotionen

Um spezifische Emotionen wie "Misstrauen", "Verlegenheit" etc. zu erkennen,
müsstest du:

1. **Eigenes Dataset erstellen** mit diesen Emotionen
2. **Modell fine-tunen** auf deinem Dataset
3. **Oder**: Ein kommerzielles API verwenden (z.B. Azure Face API, AWS Rekognition)

## Vergleich: Alt vs Neu

| Feature | Original | Verbessert |
|---------|----------|------------|
| Gesichtspunkte | 6 | 468 |
| Face Detection | BlazeFace | MediaPipe |
| Emotions-Modell | Basis | Verbessert |
| Anzeigemodi | 1 | 2 |
| Steuerung | Nur q | q + l |

## Performance

- Die verbesserte Version ist etwas langsamer (mehr Berechnungen)
- Erwarte ~15-25 FPS (statt ~30 FPS)
- Bei schwacher Hardware: Verwende die Original-Version

## Tipps

- Gute Beleuchtung verbessert die Erkennung
- Frontale Gesichter werden am besten erkannt
- Bei mehreren Gesichtern: Näher an der Kamera = bessere Erkennung
