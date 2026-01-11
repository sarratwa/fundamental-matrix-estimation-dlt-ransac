# Schätzung der Fundamentalmatrix

The goal is to compute the epipolar geometry between two uncalibrated images and estimate the
Fundamental Matrix using manually implemented algorithms. Lernziele: 
- Verständnis der Epipolargeometrie
- Umsetzung einfacher Corner Detectoren
- Feature Matching mit Deskriptoren
- DLT Algorithmus (mit Normalisierung)
- RANSAC for Model Estimation
- Visualisierung von Epipolarlinien

## Aufgabenstellung

Umsetzung der folgenden Schritte:

### Corner Detection
* Eigene Implementierung eines Corner Detectors (Shi–Tomasi).

### Feature Description & Matching
* Patch-basierter Deskriptor (11×11)
* Matching via SSD (Sum of Squared Differences)
* Kein ORB / kein BFMatcher.

### Relaxation
* Filterung der Korrespondenzen anhand ihrer Disparität.

### Berechnung der Fundamentalmatrix F
* Implementierung des DLT bzw. Normalized DLT Algorithmus
* vollständig mit NumPy (kein cv2.findFundamentalMat).

### RANSAC
* Eigener RANSAC-Loop zur Ausreißerunterdrückung
* Bucketing-Technik optional.

### Finale Schätzung von F
* DLT neu berechnet mit allen RANSAC-Inliern.

### Visualisierung
* Darstellung der Korrespondenzen
* Epipolarlinien in beiden Bildern.

## How to run 

```
python3 src/main.py
```

### Dependencies

```
numpy
matplotlib
opencv-python  
```