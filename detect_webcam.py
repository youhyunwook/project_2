import cv2
import os
import tkinter as tk
from PIL import Image, ImageTk
from ultralytics import YOLO
import time
import numpy as np

# YOLO 모델 로드
model = YOLO("best.pt")

# YOLO 모델에 정의된 클래스 이름
class_names = model.names

# 이미지 저장할 폴더
output_folder = "damaged_images"
os.makedirs(output_folder, exist_ok=True)

# Emboss 필터 커널
femboss = np.array([[-1.0, 0.0, 0.0],
                    [ 0.0, 0.0, 0.0],
                    [ 0.0, 0.0, 1.0]])

# Tkinter 창 생성
window = tk.Tk()
window.geometry("720x720")
window.title("YOLO 객체 탐지 및 필터 적용")

# 카메라 캡처
cap = cv2.VideoCapture(0)

# 상태 텍스트
status_label = tk.Label(window, text="상태: 적합", font=("Helvetica", 16, "bold"), fg="green")
status_label.pack(pady=10)

# 프레임
video_label = tk.Label(window)
video_label.pack()

# FPS 측정을 위한 초기 시간 변수
prev_time = 0

# damaged_detected가 True로 유지된 시간을 추적할 변수
damaged_start_time = None
damaged_delay = 1.0  # 1초 이상 유지되어야 부적합으로 변경
last_saved_time = 0  # 마지막 저장 시간을 추적할 변수

# 프레임 업데이트 상태 (실행/정지)
is_running = False
filter_type = "none"  # 초기 상태는 필터 없음
after_id = None

# CLAHE 적용 함수
def apply_clahe(frame):
    img_yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    img_yuv[:, :, 0] = clahe.apply(img_yuv[:, :, 0])
    img_clahe = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    return img_clahe

# Emboss 필터 적용 함수
def apply_emboss(frame):
    img_yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    img_yuv[:, :, 0] = clahe.apply(img_yuv[:, :, 0])
    img_clahe = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    gray = cv2.cvtColor(img_clahe, cv2.COLOR_BGR2GRAY)
    gray16 = np.int16(gray)
    embossed = np.uint8(np.clip(cv2.filter2D(gray16, -1, femboss) + 128, 0, 255))
    embossed = np.stack([embossed] * 3, axis=-1)
    return embossed

# 엣지 검출 필터 적용 함수
def apply_edge_detection(frame):
    img_yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    img_yuv[:, :, 0] = clahe.apply(img_yuv[:, :, 0])
    img_clahe = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    edges = cv2.Canny(img_clahe, 100, 200, 3)
    edges = np.stack([edges] * 3, axis=-1)
    return edges

def update_frame():
    global prev_time, damaged_start_time, after_id, last_saved_time, filter_type
    ret, frame = cap.read()
    if not ret:
        print("웹캠에서 영상을 가져올 수 없습니다.")
        return

    frame = cv2.flip(frame, 1)

    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    if filter_type == "clahe":
        filtered_frame = apply_clahe(frame)
    elif filter_type == "emboss":
        filtered_frame = apply_emboss(frame)
    elif filter_type == "edge":
        filtered_frame = apply_edge_detection(frame)
    else:
        filtered_frame = frame

    if is_running:
        results = model(filtered_frame)
        damaged_detected = False
        labels_to_save = []

        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                width = x2 - x1
                height = y2 - y1

                h, w, _ = frame.shape
                normalized_center_x = center_x / w
                normalized_center_y = center_y / h
                normalized_width = width / w
                normalized_height = height / h

                labels_to_save.append(f"{class_id} {normalized_center_x} {normalized_center_y} {normalized_width} {normalized_height}\n")

                if class_names[class_id] == "damaged":
                    damaged_detected = True

        if damaged_detected:
            if damaged_start_time is None:
                damaged_start_time = time.time()
            elif time.time() - damaged_start_time >= damaged_delay:
                status_label.config(text="상태: 부적합", fg="red")

                if time.time() - last_saved_time >= 5:
                    timestamp = time.strftime("%Y%m%d-%H%M%S")
                    base_filename = os.path.join(output_folder, f"damaged_{timestamp}")
                    image_path = f"{base_filename}.jpg"
                    cv2.imwrite(image_path, frame)

                    label_path = f"{base_filename}.txt"
                    with open(label_path, "w") as label_file:
                        label_file.writelines(labels_to_save)

                    last_saved_time = time.time()
        else:
            damaged_start_time = None
            status_label.config(text="상태: 적합", fg="green")

        annotated_frame = results[0].plot()
    else:
        annotated_frame = filtered_frame

    img = Image.fromarray(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
    imgtk = ImageTk.PhotoImage(image=img)
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)

    after_id = video_label.after(10, update_frame)

def toggle_running():
    global is_running
    if is_running:
        is_running = False
        toggle_button.config(text="실행")
    else:
        is_running = True
        toggle_button.config(text="정지")

def exit_program():
    cap.release()
    window.quit()

def set_filter(new_filter_type):
    global filter_type
    filter_type = new_filter_type

# 필터 버튼 중앙 정렬
filter_frame = tk.Frame(window)
filter_frame.pack(pady=10)

# 필터 버튼 생성 (가로로 나열)
clahe_button = tk.Button(filter_frame, text="CLAHE", font=("Helvetica", 14), command=lambda: set_filter("clahe"))
clahe_button.pack(side="left", padx=10)

emboss_button = tk.Button(filter_frame, text="Emboss", font=("Helvetica", 14), command=lambda: set_filter("emboss"))
emboss_button.pack(side="left", padx=10)

edge_button = tk.Button(filter_frame, text="Edge", font=("Helvetica", 14), command=lambda: set_filter("edge"))
edge_button.pack(side="left", padx=10)

# 기본 필터로 돌아가는 버튼
reset_button = tk.Button(filter_frame, text="None", font=("Helvetica", 14), command=lambda: set_filter("none"))
reset_button.pack(side="left", padx=10)    

# 버튼들을 담을 프레임 생성
button_frame = tk.Frame(window)
button_frame.pack(pady=20)

# 실행/정지 버튼 생성
toggle_button = tk.Button(button_frame, text="실행", font=("Helvetica", 20), command=toggle_running)
toggle_button.pack(side="left", padx=20)

# 종료 버튼 생성
exit_button = tk.Button(button_frame, text="종료", font=("Helvetica", 20), command=exit_program)
exit_button.pack(side="left", padx=20)

# Tkinter 메인 루프 실행
update_frame()
window.mainloop()

# 웹캠 해제
cap.release()
