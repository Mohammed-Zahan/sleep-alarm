# 😴 Sleep Alarm

> Eyes closed too long? It screams. Powered by MediaPipe FaceMesh, OpenCV, and the EAR algorithm.

A real-time drowsiness detector that watches your eyes through your webcam while you study or work. The moment your eyes stay closed for too long, an alarm fires to snap you back awake.

---

## How it works

Every frame from your webcam is run through MediaPipe FaceMesh, which maps 478 landmarks across your face. Six points around each eye are extracted and fed into the **Eye Aspect Ratio (EAR)** formula:

```
EAR = (‖p2−p6‖ + ‖p3−p5‖) / (2 × ‖p1−p4‖)
```

When your eyes are open, EAR hovers around `0.25–0.35`. When they close, it drops below `0.21`. If it stays below the threshold for 20 consecutive frames (~0.7 seconds), the alarm triggers.

---

## Tech stack

| Tool | Purpose |
|------|---------|
| Python 3.11 | Core language |
| OpenCV | Webcam feed + drawing HUD |
| MediaPipe | AI face & eye landmark detection |
| NumPy | Coordinate math |
| SciPy | Euclidean distance for EAR |
| Pygame | Playing the alarm sound |

---

### Live controls

| Key | Action |
|-----|--------|
| `Q` | Quit |
| `R` | Reset closed-eye counter |
| `+` | Raise EAR threshold (more sensitive) |
| `-` | Lower EAR threshold (less sensitive) |

---

## Requirements

- Python 3.11 (MediaPipe does not support Python 3.13+)
- A working webcam
- A speaker
