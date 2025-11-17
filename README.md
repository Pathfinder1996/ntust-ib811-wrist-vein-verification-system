## NTUST-IB811 Wrist Vein Verification System on Raspberry Pi
This repository provides the open-source implementation of the wrist vein verification system used in our journal publication.

The NTUST-IB811 Wrist Vein Dataset, collected using our NIR imaging device, can be downloaded from:

🔗 [https://ieee-dataport.org/documents/ntust-ib811-wrist-vein-dataset](https://ieee-dataport.org/documents/ntust-ib811-wrist-vein-dataset)

This system integrates wrist image acquisition, region of interest (ROI) extraction, vein enhancement, and deep-learning-based feature matching into a Raspberry Pi environment. A lightweight graphical user interface (GUI) is included for easy operation and demonstration.

## Contents
- `main.py` - GUI for the NTUST-IB811 wrist vein verification system.
- `vein_enhance.py` - Functions for vein enhancement.
- `wrist_roi.py` - Functions for wrist ROI extraction.
- `requirements.txt` - Python 3.9.2 dependency list used on Raspberry Pi.
- `Ours_model_fold_3.tflite` - The best-performing model for feature matching.

## Development on Personal Computer
The system modules were first developed and validated on a PC before deployment to the Raspberry Pi.
- ROI Extraction: [https://github.com/Pathfinder1996/wrist-roi-extraction](https://github.com/Pathfinder1996/wrist-roi-extraction)
- Vein Enhancement: [https://github.com/Pathfinder1996/biometric-vein-enhancement](https://github.com/Pathfinder1996/biometric-vein-enhancement)
- Lightweight Siamese Network Feature Matching Model (Training on PC): [https://github.com/Pathfinder1996/lightweight-hybrid-siamese-neural-network](https://github.com/Pathfinder1996/lightweight-hybrid-siamese-neural-network)

## System Workflow
- The verification pipeline consists of four stages:
1. Capture the wrist vein image
2. Extract the ROI
3. Enhance the vein image
4. Load the trained model for feature matching
   - Users may register or authenticate
   - For authentication, the system extracts the user’s vein features and compares them with the claimed identity stored in the database

- System Flowchart:

![System Flowchart](image/fig1.png)

## Example Results (Click the thumbnails to enlarge)
| Capture Wrist | Feature Extraction (ROI + Vein Enhancement) | User Registration | Feature Matching (Intentionally tested with different wrist → Rejected) |
|-------------|-----------------|-----------------|-----------------|
| ![1](image/1.png) | ![2](image/2.png) |![3](image/3.png) |![4](image/4.png) |

This demo showcases the full workflow of our wrist vein verification system on Raspberry Pi.

## Raspberry Pi OS Version
```
Debian 12 Bookworm
```

## How to Use
Install the required Python 3.9.2 packages:
```
pip install -r requirements.txt
```
Note: Some packages listed in requirements.txt cannot be installed directly via pip on Raspberry Pi.
For these modules, you must manually download the specified version (source archive or wheel file) from the official website or GitHub release page and install them locally.

Update all file paths inside `main.py` according to your system, then run the GUI:
```
python main.py
```
