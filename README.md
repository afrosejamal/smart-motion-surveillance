# 🎥 Smart Motion Surveillance

### *It doesn't just see movement — it understands it.*

Most motion-detection scripts panic at a flickering light or a moving shadow. This one doesn't. It only raises an alert when **real motion overlaps with a real object** — built on a three-layer pipeline of background subtraction, YOLOv8 object detection, and a hand-built multi-object tracker.

📸 *Demo screenshots/video coming soon*

---

## 🧠 The Core Idea

```
   Motion detected?  ──┐
                        ├──►  Overlap?  ──►  ✅ Log it (with cooldown)
   Object detected?  ──┘                ──►  ❌ Ignore the noise
```

A curtain blowing in the wind is *motion* but not an *object*. A parked car is an *object* but not *motion*. This system only cares about the intersection of both — which is what actually matters in real surveillance.

---

## ⚙️ Three Layers, Working Together

| Layer | What it does | How |
|---|---|---|
| 🌀 **Motion Layer** | Flags *where* something changed | MOG2 background subtraction + contour filtering |
| 🎯 **Detection Layer** | Identifies *what* it is | YOLOv8 (person, car, bus...) + Haar Cascade for faces |
| 🔢 **Tracking Layer** | Remembers *who's who* across frames | Custom centroid tracker — built from scratch, no external library |

The tracker is the part I'm most proud of — it assigns persistent IDs, tolerates brief disappearances (occlusion), and enforces a per-object screenshot cooldown so the same person walking past doesn't flood your folder with 40 nearly-identical images.

---

## ✨ Features at a Glance

- 🧠 **Smart triggering** — overlap-based logging, not raw pixel-change spam
- 🆔 **Persistent object IDs** — custom centroid tracker, not a black-box library
- ⏱️ **Per-object cooldown** — one object, one sane number of screenshots
- 🖼️ **Dual capture** — full scene *and* a clean, padded crop of the object
- 📊 **Live HUD overlay** — motion score, object count, FPS, status, rendered in-frame
- ⌨️ **Manual override** — force a screenshot or toggle overlays without restarting

---

## 🛠️ Built With

`OpenCV` · `Ultralytics YOLOv8` · `PyTorch` · `NumPy`

---

## 🚀 Quickstart

```bash
git clone https://github.com/afrosejamal/smart-motion-surveillance.git
cd smart-motion-surveillance
pip install -r requirements.txt
python movement_analysis.py
```

YOLOv8n weights download automatically on first run — no manual setup needed.

| Key | Action |
|---|---|
| `q` | Quit |
| `s` | Manual screenshot |
| `d` | Toggle detection overlay |

---

## 🔍 Under the Hood

1. Each frame is blurred and run through `MOG2` background subtraction to isolate movement
2. Contours above a minimum area become motion candidates
3. The custom `CentroidTracker` matches these to existing tracked IDs by nearest distance — new ID if unmatched, forgotten after too many missed frames
4. YOLOv8 runs in parallel, classifying everything in frame
5. A screenshot fires only when a tracked centroid sits inside a motion region **and** overlaps a YOLO box by >20% of its area — and only if that object's cooldown has expired

---

## 🔐 Privacy & Ethical Use

Built as a computer vision learning project — not a tool for unauthorized surveillance.

- Only run this where you own the space or have clear consent to monitor
- Check your local laws around recording people
- 100% local — no cloud upload, no external data transmission

---

## ⚠️ Known Limitations

- Fixed single-camera angle, no PTZ/multi-camera support
- Tracker can lose identity during fast motion or heavy occlusion
- CPU-only inference is noticeably slower than CUDA-enabled GPU

## 🔮 Roadmap

- [ ] Add demo video/screenshots
- [ ] Push alerts (email/notification) on detection
- [ ] Swap in DeepSORT for stronger occlusion handling
- [ ] Configurable "ignore zones" within the frame

---

## 👤 Author

**Afrose Fathima J**
📧 afrosepvt@gmail.com · 🔗 [LinkedIn](http://www.linkedin.com/in/afrose-fathima-jamal-492b57291)

⭐ *If this caught your eye, a star helps it reach more people.*
