## 2025/11/30
如果您是透過我的碩士論文連結前來，請注意本系統的程式碼與實驗結果已進行以下更新：

- 第 3.2 章節之靜脈特徵增強流程的參數全面修正：
原始論文版本在影像增強階段的部分參數設定不正確。經修正後重新訓練模型（詳見下方開發連結），在 FYO 與 PUT 測試集上的 EER 明顯下降。

- 第 4.2 章節之靜脈特徵匹配模型子網路架構更新：
現行系統採用新版的特徵匹配模型（詳見下方開發連結），相較於原論文版本，計算量已進一步減少，在不犧牲準確率的前提下提升推論效率。

- 第 5.3.4 章節之樹莓派端延遲大幅下降：
依據 `gui_test.log` 的最新 profiling 結果，目前系統各階段的平均延遲均顯著低於論文所報告的數值。本次更新移除了多項冗餘計算，並重新規劃資料流程，使整體效能大幅提升。未來版本正以編譯型語言重寫部分模組，進一步降低延遲。

## NTUST-IB811 Wrist Vein Verification System on Raspberry Pi
This repository provides the open-source implementation of the wrist vein verification system used in our journal publication.

The NTUST-IB811 Wrist Vein Dataset, collected using our NIR imaging device, can be downloaded from:

🔗 [https://ieee-dataport.org/documents/ntust-ib811-wrist-vein-dataset](https://ieee-dataport.org/documents/ntust-ib811-wrist-vein-dataset)

This system integrates wrist image acquisition, region of interest (ROI) extraction, vein enhancement, and deep-learning-based feature matching into a Raspberry Pi environment. A lightweight graphical user interface (GUI) is included for easy operation and demonstration.

## Contents
- `gui_test.log` - Console output log showing the execution timeline, camera restart behavior, capture errors, and processing latency for each system stage.
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
| Capture Wrist | Feature Extraction (ROI + Vein Enhancement) | User Registration | Feature Matching (Same wrist used for authentication → Accepted) | Execution Time Overview |
|-------------|-----------------|-----------------|-----------------|-----------------|
| ![1](image/1.png) | ![2](image/2.png) |![3](image/3.png) |![4](image/4.png) |![5](image/5.png) |

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
