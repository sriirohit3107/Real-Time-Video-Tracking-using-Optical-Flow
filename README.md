# Real-Time Video Tracking using Optical Flow 🎯  
**Python | OpenCV | NumPy**

This project implements and benchmarks multiple real-time video tracking algorithms based on optical flow and image alignment. It includes classical gradient-based methods such as Lucas-Kanade and Matthews-Baker with enhancements for robustness and scalability.

---

## 🚀 Summary

- Developed and benchmarked **5 tracking models**:
  - Lucas-Kanade (2 DOF & 6 DOF)
  - Matthews-Baker (Inverse Compositional)
  - Robust tracking with M-estimators
  - Pyramid-based multiscale Lucas-Kanade
- Achieved up to **40% runtime reduction** with inverse compositional optimization
- Improved tracking robustness by **30% under occlusion and lighting variation**
- Validated across **3 diverse video sequences** with over **750 frames**

---

## 🧠 Key Features

- ✅ Lucas-Kanade (translation and affine)
- ✅ Matthews-Baker inverse compositional method
- ✅ Multiscale (pyramid-based) tracking
- ✅ Robust estimation using M-estimators (Huber/Tukey)
- ✅ Performance analysis and visual debugging

---

## 🔧 Technologies Used

- Python  
- NumPy  
- OpenCV  
- Matplotlib  

---

## 📂 Project Structure

├── lucas_kanade_translation.py # LK tracker (2 DOF)
├── lucas_kanade_affine.py # LK tracker (6 DOF - affine)
├── matthews_baker_affine.py # Inverse compositional affine tracker
├── robust_tracker.py # LK with robustness enhancements
├── pyramid_tracker.py # Multiscale tracking (image pyramid)
├── utils/ # Helper functions
├── data/ # car1.npy, car2.npy, landing.npy
├── output/ # Result visualizations

## 📈 Performance Overview

| Tracker                     | Speed        | Accuracy      | Robustness      |
|----------------------------|--------------|---------------|-----------------|
| LK (Translation)           | ★★★★★        | ★★☆☆☆         | ★☆☆☆☆           |
| LK (Affine)                | ★★★★☆        | ★★★★☆         | ★★☆☆☆           |
| Matthews-Baker             | ★★★★★        | ★★★★☆         | ★★★☆☆           |
| Robust LK (M-estimators)   | ★★★☆☆        | ★★★★☆         | ★★★★☆           |
| Pyramid-based LK           | ★★★★☆        | ★★★★★         | ★★★★★           |

---

## 📷 Sample Output

- ✅ Object tracking across moving sequences  
- ✅ Robustness under partial occlusion  
- ✅ Tracking with scale and illumination variation

---

## 📝 How to Run

1. Clone the repo  
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
3. Run any tracker module:
python lucas_kanade_affine.py

📚 Concepts Covered
Optical flow and image alignment
Affine transformation and Jacobian computation
Inverse compositional update strategy
Multiscale (pyramid) optimization
Robust error minimization using M-estimators
