import os
import sys
import time
import threading

import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
from time import sleep
from PIL import Image
from gpiozero import LED
from PyQt5 import QtWidgets
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QInputDialog, QMessageBox
from PyQt5.QtCore import QTimer

from wrist_roi import line_intersection, scale_point
from vein_enhance import automatic_gamma_correction

# global variables for vein ehancement
biasX = 0
biasY = 0
scaleFactor = 2.0
gammaValue = 1.0
laplacianDelta = 0.0
roiMargin = 0.1
guideCircleRadiusFactor = 0.3
lineThickness = 2
circleRadian = 6
lineColor = (120, 120, 120)
circleColor = (100, 100, 100)
whiteArea_threshold = 0.75
roiShape_threshold = 0.7

# GUI initialization
app = QApplication(sys.argv)
main_window = QMainWindow()
main_window.setObjectName("MainWindow")
main_window.setWindowTitle("NTUST-IB811 Wrist Vein Verification System")
main_window.resize(1200, 680)

label = QLabel(main_window)
label.setGeometry(20, 20, 642, 480)

roi_label = QLabel(main_window)
roi_label.setGeometry(682, 20, 256, 256)

prediction_label = QLabel(main_window)
prediction_label.setGeometry(682, 296, 256, 256)

capture_button = QPushButton('Capture Wrist', main_window)
capture_button.setGeometry(958, 20, 180, 70)

recapture_button = QPushButton('Restart Camera', main_window)
recapture_button.setGeometry(958, 130, 180, 70)

seg_button = QPushButton('Feature Extraction', main_window)
seg_button.setGeometry(958, 240, 180, 70)

match_button = QPushButton('Feature Matching', main_window)
match_button.setGeometry(958, 350, 180, 70)

sign_in_button = QPushButton('User Registration', main_window)
sign_in_button.setGeometry(958, 460, 180, 70)

dataset_button = QPushButton('Dataset Collection', main_window)
dataset_button.setGeometry(958, 570, 180, 70)

ocv = True

def close_event(event):
    global ocv
    ocv = False
    cap.release()

main_window.closeEvent = close_event

cap = cv2.VideoCapture(0)

def opencv():
    global ocv, cap
    if not cap.isOpened():
        print("[Error] Camera failed")
        sys.exit()
    while ocv:
        ret, frame = cap.read()
        if not ret:
            print("[Error] Failed to capture image")
            break
        frame = cv2.resize(frame, (642, 480))
        frame_cropped = frame[:, 1:641 ]
        frame = cv2.cvtColor(frame_cropped, cv2.COLOR_BGR2RGB)
        img = QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
        label.setPixmap(QPixmap.fromImage(img))
        QApplication.processEvents()

video = threading.Thread(target=opencv)
video.start()

def capture_image():
    global cap, ocv
    if not cap.isOpened():
        print("[Error] Camera failed")
        return
    ret, frame = cap.read()

    if not ret:
        print("[Error] Failed to capture image")
        return

    frame = cv2.resize(frame, (642, 480))
    frame_cropped = frame[:, 1:641 ]
    frame = cv2.cvtColor(frame_cropped, cv2.COLOR_BGR2RGB)
    img = QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
    label.setPixmap(QPixmap.fromImage(img))

    cap_start_time = time.perf_counter()
    save_path = "captured_img.png"
    cv2.imwrite(save_path, frame)

    ocv = False
    cap_end_time = time.perf_counter()
    total_cap_time = cap_end_time - cap_start_time
    print(f"[Time] Capture time: {total_cap_time:.6f} s")

capture_button.clicked.connect(capture_image)

def restart_camera():
    global cap, ocv
    if not ocv:
        cap.release()
        cap = cv2.VideoCapture(0)
        ocv = True
        threading.Thread(target=opencv).start()
    print("[Info] Camera restarted")

recapture_button.clicked.connect(restart_camera)

def process_image():
    # compute ROI and vein enhancement, then show in roi_label/prediction_label
    roi_start_time = time.perf_counter()
    captured_img = cv2.imread('captured_img.png', 0)
    if captured_img is None:
        print("[Error] No captured image found.")
        return

    h, w = captured_img.shape
    img = np.zeros((h + 160, w + 160), np.uint8)
    img[80:-80, 80:-80] = captured_img

    # otsu binarization
    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # extract contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("[Error] No contour found.")
        return

    max_contour = max(contours, key=cv2.contourArea)
    canvas = np.zeros_like(thresh)
    cv2.drawContours(canvas, [max_contour], -1, (255), thickness=cv2.FILLED)

    cnt, _ = cv2.findContours(canvas, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cnt = cnt[0]
    img_c = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    M = cv2.moments(cnt)
    if M["m00"] != 0:
        x_c = int(M["m10"] / M["m00"])
    else:
        print("[Warning] Zero division in moments calculation")
        return

    hull = cv2.convexHull(cnt, returnPoints=False)
    defects = cv2.convexityDefects(cnt, hull)
    if defects is None:
        print("[Error] No convexity defects found.")
        return
    sorted_defects = sorted(defects, key=lambda x: x[0][3], reverse=True)

    # first defect
    first_defect = sorted_defects[0][0]
    s, e, f, _ = first_defect
    P1, P3, P4 = tuple(cnt[f][0]), tuple(cnt[s][0]), tuple(cnt[e][0])
    if P4[1] < P1[1]:
        P4, P3 = P3, P4
    is_right = P1[0] > x_c

    # second defect on the opposite side of centroid
    second_defect = None
    for defect in sorted_defects[1:]:
        s, e, f, _ = defect[0]
        far_point = tuple(cnt[f][0])
        if (far_point[0] > x_c) != is_right:
            second_defect = defect
            break
    if second_defect is None:
        print("[Error] Cannot find second defect.")
        return

    s, e, f, _ = second_defect[0]
    P2, P5, P6 = tuple(cnt[f][0]), tuple(cnt[s][0]), tuple(cnt[e][0])
    if P5[1] < P2[1]:
        P6, P5 = P5, P6

    # convert to np arrays
    P1, P2, P3, P4, P5, P6 = map(np.array, [P1, P2, P3, P4, P5, P6])

    # perpendicular line across P1
    vec_P1_P4 = P4 - P1
    vec_P1_P4_perp = np.array([-vec_P1_P4[1], vec_P1_P4[0]])
    unit_perp = vec_P1_P4_perp / np.linalg.norm(vec_P1_P4_perp)
    start_line = (P1 - unit_perp * max(h, w)).astype(int)
    end_line = (P1 + unit_perp * max(h, w)).astype(int)

    # P7 along P1 to P4 direction
    unit_vec_P1_P4 = vec_P1_P4 / np.linalg.norm(vec_P1_P4)
    P7 = (P1 + unit_vec_P1_P4 * 200).astype(int)

    # P8 intersection of the perpendicular and line (P2 to P5)
    P8 = np.array(line_intersection(start_line, end_line, P2, P5))

    # P9 extend from P8 towards P5
    vec_P8_P5 = P5 - P8
    unit_vec_P8_P5 = vec_P8_P5 / np.linalg.norm(vec_P8_P5)
    P9 = (P8 + unit_vec_P8_P5 * 200).astype(int)

    # orientation for ROI vertex ordering
    cross_z = np.cross(P8 - P1, P3 - P1)
    sin_theta = cross_z / (np.linalg.norm(P8 - P1) * np.linalg.norm(P3 - P1))

    # scale the four ROI corners about their centroid
    ROI_center = (P1 + P7 + P8 + P9) / 4
    scale = 0.8
    P1_s, P7_s, P8_s, P9_s = [scale_point(P, ROI_center, scale) for P in (P1, P7, P8, P9)]

    # order of the 4 points for perspective transform
    if sin_theta < 0:
        ROI_points = np.float32([P1_s, P8_s, P9_s, P7_s])
    else:
        ROI_points = np.float32([P8_s, P1_s, P7_s, P9_s])

    ROI_w = int(max(np.linalg.norm(P8_s - P1_s), np.linalg.norm(P9_s - P7_s)))
    ROI_h = int(max(np.linalg.norm(P8_s - P9_s), np.linalg.norm(P1_s - P7_s)))
    dst_points = np.float32([[0, 0], [ROI_w, 0], [ROI_w, ROI_h], [0, ROI_h]])

    m_persp = cv2.getPerspectiveTransform(ROI_points, dst_points)
    warped = cv2.warpPerspective(img, m_persp, (ROI_w, ROI_h))
    resized_roi = cv2.resize(warped, (64, 64))
    cv2.imwrite('wrist_extraction_roi.png', resized_roi)

    roi_end_time = time.perf_counter()
    print(f"[Time] ROI extraction time: {roi_end_time - roi_start_time:.6f} s")

    img_defects = img_c.copy()
    cv2.circle(img_defects, P1, 5, (0, 0, 255), -1)
    cv2.putText(img_defects, "P1", (P1[0] + 10, P1[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.circle(img_defects, P3, 5, (0, 0, 255), -1)
    cv2.putText(img_defects, "P3", (P3[0] + 10, P3[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.circle(img_defects, P4, 5, (0, 0, 255), -1)
    cv2.putText(img_defects, "P4", (P4[0] + 10, P4[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.circle(img_defects, P2, 5, (0, 0, 255), -1)
    cv2.putText(img_defects, "P2", (P2[0] + 10, P2[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.circle(img_defects, P5, 5, (0, 0, 255), -1)
    cv2.putText(img_defects, "P5", (P5[0] + 10, P5[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.circle(img_defects, P6, 5, (0, 0, 255), -1)
    cv2.putText(img_defects, "P6", (P6[0] + 10, P6[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.arrowedLine(img_defects, tuple(P1), tuple(P4), (255, 0, 0), 2, tipLength=0.05)
    cv2.arrowedLine(img_defects, tuple(P2), tuple(P5), (255, 0, 0), 2, tipLength=0.05)
    cv2.arrowedLine(img_defects, tuple(start_line), tuple(end_line), (255, 0, 0), 2, tipLength=0.05)
    cv2.circle(img_defects, P7, 5, (0, 0, 255), -1)
    cv2.putText(img_defects, "P7", (P7[0] + 10, P7[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.circle(img_defects, P8, 5, (0, 0, 255), -1)
    cv2.putText(img_defects, "P8", (P8[0] + 10, P8[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.circle(img_defects, P9, 5, (0, 0, 255), -1)
    cv2.putText(img_defects, "P9", (P9[0] + 10, P9[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.arrowedLine(img_defects, tuple(P1), tuple(P3), (255, 255, 255), 5, tipLength=0.05)
    cv2.arrowedLine(img_defects, tuple(P1), tuple(P8), (255, 255, 255), 5, tipLength=0.05)
    cv2.imwrite('wrist_line.png', img_defects)

    img_roi = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.line(img_roi, tuple(P8_s.astype(int)), tuple(P1_s.astype(int)), (0, 255, 0), 2)
    cv2.line(img_roi, tuple(P1_s.astype(int)), tuple(P7_s.astype(int)), (0, 255, 0), 2)
    cv2.line(img_roi, tuple(P7_s.astype(int)), tuple(P9_s.astype(int)), (0, 255, 0), 2)
    cv2.line(img_roi, tuple(P9_s.astype(int)), tuple(P8_s.astype(int)), (0, 255, 0), 2)

    cv2.circle(img_roi, P1, 5, (0, 0, 255), -1)
    cv2.putText(img_roi, "P1", (P1[0] + 10, P1[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.circle(img_roi, P7, 5, (0, 0, 255), -1)
    cv2.putText(img_roi, "P7", (P7[0] + 10, P7[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.circle(img_roi, P8, 5, (0, 0, 255), -1)
    cv2.putText(img_roi, "P8", (P8[0] + 10, P8[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.circle(img_roi, P9, 5, (0, 0, 255), -1)
    cv2.putText(img_roi, "P9", (P9[0] + 10, P9[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.imwrite('wrist_roi.png', img_roi)

    # vein enhancement pipeline
    enhc_start_time = time.perf_counter()
    img_pil = Image.open("wrist_extraction_roi.png").convert('L').resize((64, 64))
    img_np = np.array(img_pil)

    # 1. Automatic Gamma Correction
    src = automatic_gamma_correction(img_np, gammaValue, True)

    # 2. CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    src = clahe.apply(src)

    # 3. convert to float64
    src = src.astype(np.float64) / 255.0

    # 4. gaussianBlur sigma=3
    src = cv2.GaussianBlur(src, (0, 0), sigmaX=3)

    # 5. laplacian (same parameters)
    src = cv2.Laplacian(src, cv2.CV_64F, ksize=1, scale=1, delta=laplacianDelta)

    # 6. clamp negative to 0
    src = np.maximum(src, 0.0)

    # 7. normalize using max(-min, max)
    lap_min = src.min()
    lap_max = src.max()
    denom = max(-lap_min, lap_max)
    scale = 255.0 / denom if denom != 0 else 1.0

    src *= scale

    # 8. convert to 8-bit
    src = np.clip(src, 0, 255).astype(np.uint8)

    cv2.imwrite("enhance_img.png", src)
    cv2.imwrite("original_roi.png", img_np)
    enhc_end_time = time.perf_counter()
    print(f"[Time] Vein enhancement time: {enhc_end_time - enhc_start_time:.6f} s")

    # show ROI and enhanced images in GUI
    img_roi_display = cv2.imread("original_roi.png", cv2.IMREAD_GRAYSCALE)
    if img_roi_display is not None:
        img_roi_display = cv2.resize(img_roi_display, (256, 256))
        h1, w1 = img_roi_display.shape
        qimg_roi = QImage(img_roi_display.data, w1, h1, w1, QImage.Format_Grayscale8)
        roi_label.setPixmap(QPixmap.fromImage(qimg_roi))

    img_pred_display = cv2.imread("enhance_img.png", cv2.IMREAD_GRAYSCALE)
    if img_pred_display is not None:
        img_pred_display = cv2.resize(img_pred_display, (256, 256))
        h2, w2 = img_pred_display.shape
        qimg_pred = QImage(img_pred_display.data, w2, h2, w2, QImage.Format_Grayscale8)
        prediction_label.setPixmap(QPixmap.fromImage(qimg_pred))

        # display ROI outline image in main preview
    img_roi_main = cv2.imread("wrist_roi.png")
    if img_roi_main is not None:
        rgb_img = cv2.cvtColor(img_roi_main, cv2.COLOR_BGR2RGB)
        qimg_main = QImage(rgb_img.data, rgb_img.shape[1], rgb_img.shape[0], QImage.Format_RGB888)
        label.setPixmap(QPixmap.fromImage(qimg_main))

seg_button.clicked.connect(process_image)

def match_veins():
    interpreter = tflite.Interpreter(model_path=r"/home/pi/final/Ours_model_fold_3.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    def load_images_from_folder(folder):
        images = {}
        for filename in os.listdir(folder):
            if filename.endswith(".png"):
                img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = cv2.resize(img, (64, 64))
                    img = np.expand_dims(img, axis=-1)
                    user_id = filename.split('.')[0]
                    images[user_id] = img
        return images

    folder1 = r"/home/pi/final"
    folder2 = r"/home/pi/final/sign_dataset/"

    user_id_input, okPressed = QInputDialog.getText(
        main_window, "User Login", "Enter user name:", QtWidgets.QLineEdit.Normal, ""
    )
    if not okPressed or user_id_input == "":
        return

    # load current enhance image
    user_image1 = None
    prediction_path = os.path.join(folder1, "enhance_img.png")
    if os.path.exists(prediction_path):
        user_image1 = cv2.imread(prediction_path, cv2.IMREAD_GRAYSCALE)
        if user_image1 is not None:
            user_image1 = cv2.resize(user_image1, (64, 64))
            user_image1 = np.expand_dims(user_image1, axis=-1)

    # load registered enhance image by id
    images2 = load_images_from_folder(folder2)
    user_image2 = images2.get(user_id_input)

    msg = QMessageBox()
    msg.setIcon(QMessageBox.Information)
    msg.setWindowTitle("Matching Result")

    def predict_with_tflite_model(image1, image2):
        interpreter.set_tensor(input_details[0]['index'], np.array([image1], dtype=np.float32))
        interpreter.set_tensor(input_details[1]['index'], np.array([image2], dtype=np.float32))
        interpreter.invoke()
        return interpreter.get_tensor(output_details[0]['index'])

    threshold = 0.36050936674794276

    if user_image1 is not None and user_image2 is not None:
        start = time.perf_counter()
        dist = predict_with_tflite_model(user_image1, user_image2)
        distance_value = float(dist[0][0])
        result = "Genuine" if distance_value < threshold else "Imposter"
        end = time.perf_counter()
        print(f"[Time] Matching time: {end - start:.6f} s")

        if result == "Genuine":
            relay = LED(18)
            relay.on()
            relay.off()
            msg.setText(
                f"User: {user_id_input}\n"
                f"Distance: {distance_value:.4f}\n"
                f"Result: {result}\n"
                f"Access Granted"
            )
        else:
            msg.setText(
                f"User: {user_id_input}\n"
                f"Distance: {distance_value:.4f}\n"
                f"Result: {result}\n"
                f"Access Denied"
            )
    else:
        msg.setText(f"User: {user_id_input} not found")
    msg.exec_()

match_button.clicked.connect(match_veins)

def sign_in_image():
    # save current enhance_img.png as a registered user template
    save_to_sign_path = r"/home/pi/final/sign_dataset"
    os.makedirs(save_to_sign_path, exist_ok=True)

    image_name, ok = QInputDialog.getText(main_window, 'User Registration', 'Enter registration name:')
    if not ok or not image_name:
        QMessageBox.warning(main_window, 'Error', 'Invalid user name')
        return

    full_save_path = os.path.join(save_to_sign_path, f'{image_name}.png')
    if os.path.exists(full_save_path):
        QMessageBox.warning(main_window, 'Warning', 'User already exists')
        return

    reply = QMessageBox.question(main_window, 'Confirm',
                                 f'Confirm registration as {image_name}?',
                                 QMessageBox.Yes | QMessageBox.No)
    if reply == QMessageBox.Yes:
        signin_path = "enhance_img.png"
        signin_image = cv2.imread(signin_path)
        if signin_image is not None:
            cv2.imwrite(full_save_path, signin_image)
            QMessageBox.information(main_window, 'Success', 'Image saved to database')
        else:
            QMessageBox.warning(main_window, 'Error', 'Failed to read image')

sign_in_button.clicked.connect(sign_in_image)

def save_dataset():
    registration_name = QtWidgets.QInputDialog.getText(None, "Enter ID", "Enter ID:")[0]
    if not registration_name:
        return

    hand = QtWidgets.QInputDialog.getItem(None, "Select Hand", "Left (L) or Right (R):", ["L", "R"])[0]
    gender = QtWidgets.QInputDialog.getItem(None, "Select Gender", "Male (M) or Female (F):", ["M", "F"])[0]
    session = QtWidgets.QInputDialog.getItem(None, "Select Session", "Session1 (S1) or Session2 (S2):", ["S1", "S2"])[0]

    folder_path = r"/home/pi/final/wrist_dataset/"
    os.makedirs(folder_path, exist_ok=True)

    for i in range(10):
        ret, frame = cap.read()
        if not ret:
            print("[Error] Failed to capture frame")
            return

        frame = cv2.resize(frame, (642, 480))
        frame_cropped = frame[:, 1:641 ]

        file_name = f"{registration_name}_{hand}_{gender}_{session}_{i+1:02d}.png"
        save_path = os.path.join(folder_path, file_name)

        if os.path.exists(save_path):
            QMessageBox.warning(main_window, 'Warning', 'This ID already exists file, aborting')
            save_dataset()
            return

        cv2.imwrite(save_path, frame_cropped)
        QApplication.processEvents()
        time.sleep(0.2)
    print("[Info] Dataset collection completed")

dataset_button.clicked.connect(save_dataset)

# run app
main_window.show()
sys.exit(app.exec_())
