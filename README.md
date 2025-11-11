# 🎥 Video Motion Attenuation Using Steerable Pyramids

**Author:** Yue Xu   
**Date:** June 3, 2020  
**Contact:** yxu7@caltech.edu

---

## 📘 Overview

This project implements a **phase-based motion attenuation algorithm** for videos using **complex-valued steerable pyramids**, following  
Wadhwa et al. (2013), *Phase-Based Video Motion Processing* (ACM Trans. Graph. 32(4):80).

The goal is to reduce small, unwanted motions (like vibration or drift) while preserving the main structure of the video.

---

## ⚙️ Method Summary

The algorithm works in three steps:

1. **Steerable Pyramid Decomposition** – Each frame is decomposed into multiple spatial scales and orientations.  
2. **Phase Median Filtering** – Phase values are smoothed across neighboring frames to suppress minor motions.  
3. **Reconstruction** – The modified pyramids are combined to form stabilized video frames.

The method supports both:
- **RGB processing** (applies to all color channels)  
- **Fast YIQ mode** (applies to luminance only for speed)

---

## 🧩 Repository Structure

```
.
├── main.py                             # Main implementation script
├── data/                               # Input videos
├── results/                            # Output videos and figures
├── project_report_final.pdf            # Detailed report
└── README.md
```

---

## ▶️ Quick Start

### 1️⃣ Install Dependencies
```bash
pip install numpy opencv-python matplotlib pillow
```

### 2️⃣ Run the Demo
```python
from main import processVideoMedianPhase

processVideoMedianPhase(
    "./data/moon.avi",
    "./results/moon_attenuated_YIQ.avi",
    window_size=11,
    fastYIQ=True,
    max_num_frames=25
)
```

### 3️⃣ Generate Example Figures
```python
from main import createReportFigures
createReportFigures()
```

---

## 🧠 Notes

- **YIQ mode** is faster with similar visual quality to RGB.  
- Larger `window_size` values remove more small motions.  
- Designed for videos like those in the `data/`: *moon.avi*, *subway.mp4*, and *fly03_short.avi*.  

---

## 📖 How to Cite

This project was completed for Caltech's CS101C - Computational Camera, taught by [Prof. Katie Bowman](https://www.cms.caltech.edu/people/klbouman) in Spring 2020. For more information about the course and my project, see this [Caltech news feature](https://thisis.caltech.edu/news/katie-bouman-cs-101-remote-learning).

Wadhwa, N., Rubinstein, M., Durand, F., & Freeman, W. (2013).  
*Phase-Based Video Motion Processing.*  
**ACM Transactions on Graphics**, 32(4), Article 80.  
[DOI:10.1145/2461912.2461966](http://doi.acm.org/10.1145/2461912.2461966)

This code is distributed for educational and research purposes only. Please cite the above reference and this Github repo if you use or extend this implementation.

The fruit fly two-photo imaging video was recorded in [Prof. Elizabeth Hong's](https://www.bbe.caltech.edu/people/elizabeth-j-hong) lab.