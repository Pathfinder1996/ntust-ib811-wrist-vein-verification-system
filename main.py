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

app = QApplication(sys.argv)
main_window = QMainWindow()
main_window.setObjectName("MainWindow")
main_window.setWindowTitle("NTUST-IB811 靜脈識別系統")
main_window.resize(1200, 680)

label = QLabel(main_window)
label.setGeometry(20, 20, 642, 480)

roi_label = QLabel(main_window)
roi_label.setGeometry(682, 20, 256, 256)

prediction_label = QLabel(main_window)
prediction_label.setGeometry(682, 296, 256, 256)

capture_button = QPushButton('拍攝手腕', main_window)
capture_button.setGeometry(958, 20, 180, 70)

recapture_button = QPushButton('重新拍攝', main_window)
recapture_button.setGeometry(958, 130, 180, 70)

seg_button = QPushButton('特徵提取', main_window)
seg_button.setGeometry(958, 240, 180, 70)

match_button = QPushButton('特徵匹配', main_window)
match_button.setGeometry(958, 350, 180, 70)

sign_in_button = QPushButton('用戶註冊', main_window)
sign_in_button.setGeometry(958, 460, 180, 70)

dataset_button = QPushButton('資料庫收集', main_window)
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
        print("無法開啟相機")
        sys.exit()
    while ocv:
        ret, frame = cap.read()
        if not ret:
            print("無法讀取畫面")
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
        print("無法開啟相機")
        return
    ret, frame = cap.read()

    if not ret:
        print("無法讀取畫面")
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

recapture_button.clicked.connect(restart_camera)

def process_image():
    roi_start_time = time.perf_counter()

    captured_img = cv2.imread('captured_img.png', 0)
    h, w = captured_img.shape
    img = np.zeros((h+160, w+160), np.uint8)
    img[80:-80, 80:-80] = captured_img
    
    # Otsu binarization
    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Extract contours and fill largest one
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    max_contour = max(contours, key=cv2.contourArea)
    canvas = np.zeros_like(thresh)
    cv2.drawContours(canvas, [max_contour], -1, (255), thickness=cv2.FILLED)

    cnt, _ = cv2.findContours(canvas, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    img_c = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cnt = cnt[0]

    M = cv2.moments(cnt)
    if M["m00"] != 0:
        x_c = int(M["m10"] / M["m00"])
        y_c = int(M["m01"] / M["m00"])
    else:
        print("[Warning] Zero division in moments calculation")

    # Find convexity defects
    hull = cv2.convexHull(cnt, returnPoints=False)
    defects = cv2.convexityDefects(cnt, hull)
    sorted_defects = sorted(defects, key=lambda x: x[0][3], reverse=True)

    # Extract the first and second maximum defects (P1, P2)
    first_defect = sorted_defects[0][0]
    s, e, f, _ = first_defect
    P1, P3, P4 = tuple(cnt[f][0]), tuple(cnt[s][0]), tuple(cnt[e][0])
    if P4[1] < P1[1]: P4, P3 = P3, P4
    is_right = P1[0] > x_c

    second_defect = None
    for defect in sorted_defects[1:]:
        s, e, f, _ = defect[0]
        far_point = tuple(cnt[f][0])
        if (far_point[0] > x_c) != is_right:
            second_defect = defect
            break

    s, e, f, _ = second_defect[0]
    P2, P5, P6 = tuple(cnt[f][0]), tuple(cnt[s][0]), tuple(cnt[e][0])
    if P5[1] < P2[1]: 
        P6, P5 = P5, P6

    # Convert points to np.array for calculations
    P1 = np.array(P1)
    P2 = np.array(P2)
    P3 = np.array(P3)
    P4 = np.array(P4)
    P5 = np.array(P5)
    P6 = np.array(P6)

    # Compute perpendicular vector
    vec_P1_P4 = P4 - P1
    vec_P1_P4_perp = np.array([-vec_P1_P4[1], vec_P1_P4[0]])
    unit_perp = vec_P1_P4_perp / np.linalg.norm(vec_P1_P4_perp)
    start_line = (P1 - unit_perp * max(h, w)).astype(int)
    end_line = (P1 + unit_perp * max(h, w)).astype(int)

    # Compute P7
    unit_vec_P1_P4 = vec_P1_P4 / np.linalg.norm(vec_P1_P4)
    P7 = (P1 + unit_vec_P1_P4 * 200).astype(int)
    
    # Compute intersection point(P8)
    P8 = np.array(line_intersection(start_line, end_line, P2, P5))

    # Compute P9
    vec_P8_P5 = P5 - P8
    unit_vec_P8_P5 = vec_P8_P5 / np.linalg.norm(vec_P8_P5)
    P9 = (P8 + unit_vec_P8_P5 * 200).astype(int)

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

    # Compute angle direction
    cross_z = np.cross(P8 - P1, P3 - P1)
    sin_theta = cross_z / (np.linalg.norm(P8 - P1) * np.linalg.norm(P3 - P1))

    # Scale points around ROI center
    ROI_center = (P1 + P7 + P8 + P9) / 4
    scale = 0.8
    P1_s = scale_point(P1, ROI_center, scale)
    P7_s = scale_point(P7, ROI_center, scale)
    P8_s = scale_point(P8, ROI_center, scale)
    P9_s = scale_point(P9, ROI_center, scale)

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
    
    # Define ROI order (based on angle)
    if sin_theta < 0:
        ROI_points = np.float32([P1_s, P8_s, P9_s, P7_s])
    else:
        ROI_points = np.float32([P8_s, P1_s, P7_s, P9_s])
        
    # Perspective transformation
    ROI_w = int(max(np.linalg.norm(P8_s - P1_s), np.linalg.norm(P9_s - P7_s)))
    ROI_h = int(max(np.linalg.norm(P8_s - P9_s), np.linalg.norm(P1_s - P7_s)))
    dst_points = np.float32([[0, 0], [ROI_w, 0], [ROI_w, ROI_h], [0, ROI_h]])

    m_persp = cv2.getPerspectiveTransform(ROI_points, dst_points)
    warped = cv2.warpPerspective(img, m_persp, (ROI_w, ROI_h))
    
    # Resize and save result
    resized_roi = cv2.resize(warped, (128, 128))

    cv2.imwrite('wrist_extraction_roi.png', resized_roi)
    roi_end_time = time.perf_counter()
    total_roi_time = roi_end_time - roi_start_time
    print(f"[Time] ROI extraction time: {total_roi_time:.6f} s")

    # Vein enhancement
    enhc_start_time = time.perf_counter()
    img = Image.open("wrist_extraction_roi.png").convert('L')
    img = img.resize((256, 256))
    img_np = np.array(img)

    # Step 1: Automatic Gamma Correction
    agc_image = automatic_gamma_correction(img_np, gamma=1.0, is_auto_mode=True)

    # Step 2: CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    clahe_image = clahe.apply(agc_image)

    # Step 3: Gaussian blur (low-pass filtering)
    normalized_image = clahe_image.astype(np.float64) / 255.0
    blurred_image = cv2.GaussianBlur(normalized_image, (0, 0), sigmaX=4)

    # Step 4: Laplacian (high-pass filtering)
    laplacian_filtered = cv2.Laplacian(
        blurred_image, 
        ddepth=cv2.CV_64F, 
        ksize=1, 
        scale=1, 
        delta=0.0, 
        borderType=cv2.BORDER_DEFAULT
    )

    laplacian_image = np.maximum(laplacian_filtered, 0.0)
    
    # Normalize output to 0~255
    lap_min, lap_max, _, _ = cv2.minMaxLoc(laplacian_image)
    scale = 255.0 / max(-lap_min, lap_max) if max(-lap_min, lap_max) != 0 else 1.0
    final_image = laplacian_image * scale
    final_image_8bit = final_image.astype(np.uint8)

    save_path1 = "Prediction_img.png"
    cv2.imwrite(save_path1, final_image_8bit)

    save_path_2 = "Original.png"
    cv2.imwrite(save_path_2, img_np) 

    enhc_end_time = time.perf_counter()
    total_enhc_time = enhc_end_time - enhc_start_time
    print(f"[Time] Vein enhancement time: {total_enhc_time:.6f} s")

    qimg2 = QImage(img_roi.data, img_roi.shape[1], img_roi.shape[0], QImage.Format_RGB888)
    label.setPixmap(QPixmap.fromImage(qimg2))

    
def segment_veins():
    process_image()

    # Read the original image
    img = cv2.imread("Original.png", cv2.IMREAD_GRAYSCALE)
    height, width = img.shape

    # Read the binary image
    save_path1 = "Prediction_img.png"
    prediction = cv2.imread(save_path1, cv2.IMREAD_GRAYSCALE)

    # Convert grayscale image to QImage and display
    qimg = QImage(img.data, width, height, width, QImage.Format_Grayscale8)
    roi_label.setPixmap(QPixmap.fromImage(qimg))

    # Convert binary image to QImage and display
    qimg_prediction = QImage(prediction.data, width, height, width, QImage.Format_Grayscale8)
    prediction_label.setPixmap(QPixmap.fromImage(qimg_prediction))

seg_button.clicked.connect(segment_veins)

def match_veins():
    interpreter = tflite.Interpreter(model_path=r"/home/pi/test/Ours_model_fold_3.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    def load_images_from_folder(folder):
        images = {}
        for filename in os.listdir(folder):
            # print("Found file:", filename)
            if filename.endswith(".png"):
                img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = cv2.resize(img, (128, 128))
                    img = np.expand_dims(img, axis=-1)
                    user_id = filename.split('.')[0]
                    images[user_id] = img
                    # print("Loaded image for user:", user_id)
        return images

    folder1 = r"/home/pi/test"
    folder2 = r"/home/pi/test/sign_dataset/"

    user_id_input, okPressed = QInputDialog.getText(main_window, "用戶登錄", "請輸入用戶名稱:", QtWidgets.QLineEdit.Normal, "")
    if okPressed and user_id_input != '':
        # print("User ID Input:", user_id_input)
        prediction_image_path = os.path.join(folder1, "Prediction_img.png")
        # print("Checking prediction image path:", prediction_image_path)

        user_image1 = None
        if os.path.exists(prediction_image_path):
            user_image1 = cv2.imread(prediction_image_path, cv2.IMREAD_GRAYSCALE)
            # print(user_image1)
            if user_image1 is not None:
                user_image1 = cv2.resize(user_image1, (128, 128))
                user_image1 = np.expand_dims(user_image1, axis=-1)

    images2 = load_images_from_folder(folder2)
    user_image2 = images2.get(user_id_input)

    msg = QMessageBox()
    msg.setIcon(QMessageBox.Information)
    msg.setWindowTitle("匹配結果")

    def predict_with_tflite_model(image1, image2):

        interpreter.set_tensor(input_details[0]['index'], np.array([image1], dtype=np.float32))
        interpreter.set_tensor(input_details[1]['index'], np.array([image2], dtype=np.float32))
        
        interpreter.invoke()

        output_data = interpreter.get_tensor(output_details[0]['index'])
        return output_data

    threshold = 0.52

    if user_image1 is not None and user_image2 is not None:

        # display_image1 = cv2.cvtColor(user_image1, cv2.COLOR_GRAY2RGB)
        # display_image2 = cv2.cvtColor(user_image2, cv2.COLOR_GRAY2RGB)

        # cv2.imshow("User Image", display_image1)
        # cv2.imshow("Database Image", display_image2)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        match_start_time = time.perf_counter()
        dist = predict_with_tflite_model(user_image1, user_image2)
        distance_value = dist[0][0]
        result = "Genuine" if distance_value < threshold else "Imposter"

        match_end_time = time.perf_counter()
        total_match_time = match_end_time - match_start_time
        print(f"[Time] Vein feature matching time: {total_match_time:.6f} s")

        if result == "Genuine":

            relay = LED(18)
            relay.on()
            # sleep(1)
            relay.off()

            msg.setText(f"與資料庫: {user_id_input}\n距離: {distance_value}\n匹配結果: {result}\n已解鎖")
            msg.show()
            QApplication.processEvents()
            QTimer.singleShot(5000, msg.close)

        else:
            msg.setText(f"與資料庫: {user_id_input}\n距離: {distance_value}\n匹配結果: {result}\n無權訪問")

    else:
        msg.setText(f"用戶: {user_id_input} 不存在")
    msg.exec_()

match_button.clicked.connect(match_veins)

def sign_in_image():
    save_to_sign_path = r"/home/pi/test/sign_dataset"

    image_name, ok = QInputDialog.getText(main_window, '用戶註冊', '請輸入註冊名稱:')
    if ok and image_name:
        full_save_path = os.path.join(save_to_sign_path, f'{image_name}.png')
        
        if os.path.exists(full_save_path):
            QMessageBox.warning(main_window, '警告', '用戶已存在，請重新輸入註冊名稱')
            sign_in_image()
            return

        reply = QMessageBox.question(main_window, '注意', f'確認註冊為 {image_name} 嗎？', QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            signin_path = "Prediction_img.png"
            signin_image = cv2.imread(signin_path)
            if sign_in_image is not None:
                cv2.imwrite(full_save_path, sign_in_image)
                QMessageBox.information(main_window, '圖像保存', f'圖像已保存到資料庫')
            else:
                QMessageBox.warning(main_window, '錯誤', '讀取圖像失敗，請重新拍攝')
        else:
            return
    else:
        QMessageBox.warning(main_window, '錯誤', '請輸入有效的用戶名稱')

sign_in_button.clicked.connect(sign_in_image)

def save_dataset():
    global cap
    if not cap.isOpened():
        print("無法開啟相機")
        return
    
    registration_name = QtWidgets.QInputDialog.getText(None, "輸入編號", "請輸入編號:")[0]
    if not registration_name:
        return

    hand = QtWidgets.QInputDialog.getItem(None, "選擇左右手", "左手(L)還是右手(R):", ["L", "R"])[0]
    gender = QtWidgets.QInputDialog.getItem(None, "選擇性別", "男(M)或女(F):", ["M", "F"])[0]
    session = QtWidgets.QInputDialog.getItem(None, "選擇時期", "時期1(S1)或時期2(S2)", ["S1", "S2"])[0]

    folder_path = r"/home/pi/test/wrist_dataset/"

    for i in range(10):
        ret, frame = cap.read()
        if not ret:
            print("無法讀取畫面")
            return

        frame = cv2.resize(frame, (642, 480))
        frame_cropped = frame[:, 1:641 ]

        file_name = f"{registration_name}_{hand}_{gender}_{session}_{i+1:02d}.png"
        save_path = os.path.join(folder_path, file_name)

        if os.path.exists(save_path):
            QMessageBox.warning(main_window, '警告', '此編號已存在，請重新輸入編號')
            save_dataset()
            return
        
        cv2.imwrite(save_path, frame_cropped)
        QApplication.processEvents() 
        time.sleep(0.2)

    print("收集完畢")

dataset_button.clicked.connect(save_dataset)

main_window.show()
sys.exit(app.exec_())
