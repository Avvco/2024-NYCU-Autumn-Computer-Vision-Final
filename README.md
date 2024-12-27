# Vanishing Point Detection

This repository contains two methods for detecting vanishing points in images:

## 1. `RANSAC_vp_detection.py`
This script implements our own method for finding vanishing points using the RANSAC algorithm combined with line detection.

## 2. `lu_vp_detection.py`
This script follows the method proposed in Xiaohu Lu's paper for detecting three orthogonal vanishing points using edge and line detection.

## How to Run
To run either of the methods, use the following commands in the terminal:

```bash
python RANSAC_vp_detection.py
```
or

```bash
python lu_vp_detection.py
```

**Note**: Make sure to modify the folder paths in the code to match your local directory structure for correct functionality.

## Python Dependencies

- lu_vp_detect
- opencv-python
- numpy