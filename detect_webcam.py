import cv2
import os
import sys
import time
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QTextEdit
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, Qt
from ultralytics import YOLO

class App(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()  # UI 초기화
        self.initVariables()  # 변수 초기화
        self.initCamera()  # 카메라 초기화
        self.prev_time = time.time()  # FPS 계산을 위한 초기 시간 변수

    def initUI(self):
        self.setWindowTitle("실시간 포장 불량 검출")  # 창 제목 설정
        self.setGeometry(100, 100, 800, 600)  # 창 크기 설정
        self.setStyleSheet("background-color: black;")  # 배경색 설정

        # 제목 레이블 추가
        self.title_label = QLabel("실시간 포장 불량 검출", self)
        self.title_label.setStyleSheet("color: white; font-size: 24px; font-weight: bold;")
        self.title_label.setAlignment(Qt.AlignCenter)

        # 비디오 표시 레이블 추가
        self.video_label = QLabel(self)
        self.video_label.setGeometry(10, 10, 640, 480)  # 위치 및 크기 설정
        self.video_label.setStyleSheet("border: 2px solid white; border-radius: 10px;")

        center_layout = QHBoxLayout()
        center_layout.addWidget(self.video_label)
        center_layout.setAlignment(self.video_label, Qt.AlignCenter)  # 가운데 정렬

        # 상태 레이블 추가
        self.status_label = QLabel("상태: 적합", self)
        self.status_label.setStyleSheet("color: green; font-size: 20px; font-weight: bold;")  # 상태 글자 스타일 설정

        # 상태 레이블을 감싸는 레이아웃 추가
        status_layout = QHBoxLayout()
        status_layout.addWidget(self.status_label)
        status_layout.setAlignment(self.status_label, Qt.AlignCenter)  # 가운데 정렬

        # 상태 텍스트를 표시하는 QTextEdit 추가
        self.status_text = QTextEdit(self)
        self.status_text.setReadOnly(True)  # 읽기 전용 설정
        self.status_text.setStyleSheet("background-color: white; border-radius: 5px; font-weight: bold;")  # 스타일 설정
        self.status_text.setFixedWidth(600)

        # 상태 텍스트를 감싸는 레이아웃 추가
        text_layout = QHBoxLayout()
        text_layout.addWidget(self.status_text)
        text_layout.setAlignment(self.status_text, Qt.AlignCenter)  # 가운데 정렬

        # 필터 버튼 추가
        filter_layout = QHBoxLayout()
        self.clahe_button = self.create_button("CLAHE", self.set_filter_clahe)
        self.emboss_button = self.create_button("Emboss", self.set_filter_emboss)
        self.edge_button = self.create_button("Edge Detection", self.set_filter_edge)
        self.none_button = self.create_button("None", self.set_filter_none)

        filter_layout.addWidget(self.clahe_button)
        filter_layout.addWidget(self.emboss_button)
        filter_layout.addWidget(self.edge_button)
        filter_layout.addWidget(self.none_button)

        # 실행 및 종료 버튼 추가
        button_layout = QHBoxLayout()
        self.toggle_button = self.create_button("실행", self.toggle_running, color="green")
        exit_button = self.create_button("종료", self.close_program, color="red")

        button_layout.addWidget(self.toggle_button)
        button_layout.addWidget(exit_button)

        # FPS 레이블 추가
        self.fps_label = QLabel(self)
        self.fps_label.setStyleSheet("color: white; font-weight: bold;")  # 스타일 설정

        # 전체 레이아웃 설정
        layout = QVBoxLayout()
        layout.addWidget(self.title_label)  # 제목 추가
        layout.addLayout(center_layout)  # 비디오 레이아웃 추가
        layout.addLayout(status_layout)  # 상태 레이블 레이아웃 추가
        layout.addLayout(filter_layout)  # 필터 버튼 레이아웃 추가
        layout.addLayout(text_layout)  # 상태 텍스트 레이아웃 추가
        layout.addLayout(button_layout)  # 실행 및 종료 버튼 레이아웃 추가
        layout.addWidget(self.fps_label)  # FPS 레이블 추가

        self.setLayout(layout)  # 최종 레이아웃 설정

    def create_button(self, text, callback, color="orange"):
        # 버튼 생성 및 설정
        button = QPushButton(text)
        button.setStyleSheet(f"background-color: {color}; font-size: 14px; border-radius: 5px;")
        button.clicked.connect(callback)  # 클릭 시 콜백 연결
        button.setFixedSize(100, 40)  # 버튼 크기 설정

        # 버튼 hover 효과 설정
        button.enterEvent = lambda event: button.setStyleSheet(f"background-color: gray; border-radius: 5px;")
        button.leaveEvent = lambda event: button.setStyleSheet(f"background-color: {color}; border-radius: 5px;")
        
        return button

    def initVariables(self):
        # YOLO 모델 초기화
        self.model = YOLO(r"C:\Users\LEE\Desktop\project\프로젝트2\detect_webcam\best.pt")
        self.class_names = self.model.names  # 클래스 이름 가져오기

        # 폴더 생성
        self.faulty_folder = "faulty"
        self.goods_folder = "goods"
        os.makedirs(self.faulty_folder, exist_ok=True)
        os.makedirs(self.goods_folder, exist_ok=True)

        # 상태 관련 변수 초기화
        self.damaged_start_time = None
        self.damaged_duration = 0
        self.damaged_threshold = 2.0
        self.status_maintenance_time = 3.0

        self.no_damaged_start_time = time.time()
        self.no_damaged_threshold = 10.0

        self.faulty_count = 0  # 불량 카운트
        self.goods_count = 0  # 양호 카운트
        self.is_running = False  # 실행 상태 플래그
        self.filter_type = "none"  # 필터 초기 설정

    def initCamera(self):
        # 카메라 초기화 및 프레임 업데이트 설정
        self.cap = cv2.VideoCapture(0)  # 웹캠 열기
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)  # 프레임 업데이트 연결
        self.timer.start(10)  # 10ms마다 업데이트

    def update_frame(self):
        # 프레임 업데이트 및 처리
        ret, frame = self.cap.read()
        if not ret:
            print("웹캠에서 영상을 가져올 수 없습니다.")
            return

        frame = cv2.flip(frame, 1)  # 프레임 좌우 반전
        filtered_frame = self.apply_filter(frame)  # 필터 적용

        if self.is_running:
            results = self.model(filtered_frame)  # 모델로부터 결과 가져오기
            self.process_results(results, frame)  # 결과 처리
            annotated_frame = results[0].plot()  # 결과를 프레임에 주석 추가
        else:
            annotated_frame = filtered_frame  # 필터링된 프레임 그대로 사용

        # FPS 계산
        current_time = time.time()
        fps = 1 / (current_time - self.prev_time)
        self.prev_time = current_time
        cv2.putText(annotated_frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # QImage로 변환하여 비디오 레이블에 표시
        img = QImage(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), annotated_frame.shape[1], annotated_frame.shape[0], QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(img))

    def apply_filter(self, frame):
        # 선택된 필터 적용
        if self.filter_type == "clahe":
            return self.apply_clahe(frame)
        elif self.filter_type == "emboss":
            return self.apply_emboss(frame)
        elif self.filter_type == "edge":
            return self.apply_edge_detection(frame)
        return frame  # 필터가 없으면 원래 프레임 반환

    def apply_clahe(self, frame):
        # CLAHE 필터 적용
        img_yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)  # YUV 색상 공간으로 변환
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))  # CLAHE 객체 생성
        img_yuv[:, :, 0] = clahe.apply(img_yuv[:, :, 0])  # 밝기 조정
        return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)  # 다시 BGR로 변환

    def apply_emboss(self, frame):
        # Emboss 필터 적용
        img_yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        img_yuv[:, :, 0] = clahe.apply(img_yuv[:, :, 0])
        img_clahe = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
        gray = cv2.cvtColor(img_clahe, cv2.COLOR_BGR2GRAY)  # 그레이스케일로 변환
        embossed = cv2.filter2D(gray, -1, np.array([[-1, 0, 0], [0, 0, 0], [0, 0, 1]])) + 128  # 엠보스 효과 적용
        return cv2.cvtColor(np.clip(embossed, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)  # 다시 BGR로 변환

    def apply_edge_detection(self, frame):
        # Edge Detection 필터 적용
        edges = cv2.Canny(frame, 100, 200)  # Canny Edge Detection
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)  # 컬러 이미지로 변환
        return edges_colored

    def process_results(self, results, frame):
        # 결과 처리 및 상태 업데이트
        damaged_detected = False  # 불량 탐지 플래그
        labels_to_save = []  # 저장할 라벨 리스트

        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])  # 클래스 ID 가져오기
                x1, y1, x2, y2 = box.xyxy[0].tolist()  # 경계 상자 좌표 가져오기
                if self.class_names[class_id] == "damaged":  # 불량 클래스 확인
                    damaged_detected = True
                h, w, _ = frame.shape
                labels_to_save.append(f"{class_id} {x1/w} {y1/h} {(x2-x1)/w} {(y2-y1)/h}\n")  # 라벨 정보 저장

        current_time = time.time()  # 현재 시간 가져오기

        if damaged_detected:
            if self.damaged_start_time is None:
                self.damaged_start_time = current_time  # 불량 탐지 시작 시간 설정
            else:
                self.damaged_duration += current_time - self.damaged_start_time  # 불량 지속 시간 증가
            if self.damaged_duration >= self.damaged_threshold:
                self.status_label.setText("상태: 부적합")  # 상태 업데이트
                self.status_label.setStyleSheet("font-size: 22px; font-weight: bold; color: red;")  # 텍스트 색상 변경
        else:
            self.damaged_start_time = None
            self.damaged_duration = 0  # 불량 탐지 초기화
            self.status_label.setText("상태: 적합")  # 상태 업데이트
            self.status_label.setStyleSheet("font-size: 22px; font-weight: bold; color: green;")  # 텍스트 색상 변경

        # 불량 탐지 상태 유지 시간 체크
        if current_time - self.no_damaged_start_time >= self.no_damaged_threshold:
            if self.damaged_duration >= self.status_maintenance_time:
                self.save_to_folder(self.faulty_folder, frame, labels_to_save)  # 불량 데이터 저장
                self.faulty_count += 1
                self.status_text.append(f"faulty 폴더에 데이터 저장: ({self.faulty_count})")
            else:
                self.save_to_folder(self.goods_folder, frame, labels_to_save)  # 양호 데이터 저장
                self.goods_count += 1
                self.status_text.append(f"goods 폴더에 데이터 저장: ({self.goods_count})")
            self.damaged_duration = 0
            self.no_damaged_start_time = current_time  # 초기화

    def save_to_folder(self, folder, frame, labels):
        # 폴더에 데이터 저장
        timestamp = time.strftime("%Y%m%d-%H%M%S")  # 현재 시간으로 타임스탬프 생성
        base_filename = os.path.join(folder, f"saved_{timestamp}")  # 저장할 파일 이름
        cv2.imwrite(f"{base_filename}.jpg", frame)  # 프레임 저장
        with open(f"{base_filename}.txt", "w") as label_file:
            label_file.writelines(labels)  # 라벨 정보 저장

    def toggle_running(self):
        # 실행 상태 토글
        self.is_running = not self.is_running
        if self.is_running:
            self.toggle_button.setText("정지")  # 버튼 텍스트 변경
            self.toggle_button.setStyleSheet("background-color: orange; font-size: 14px; border-radius: 5px;")  # 스타일 변경
        else:
            self.toggle_button.setText("실행")  # 버튼 텍스트 변경
            self.toggle_button.setStyleSheet("background-color: green; font-size: 14px; border-radius: 5px;")  # 스타일 변경

    def set_filter_clahe(self):
        self.filter_type = "clahe"  # CLAHE 필터 설정

    def set_filter_emboss(self):
        self.filter_type = "emboss"  # Emboss 필터 설정

    def set_filter_edge(self):
        self.filter_type = "edge"  # Edge Detection 필터 설정

    def set_filter_none(self):
        self.filter_type = "none"  # 필터 없음 설정

    def close_program(self):
        # 프로그램 종료 처리
        self.cap.release()  # 카메라 해제
        cv2.destroyAllWindows()  # 모든 OpenCV 윈도우 닫기
        self.close()  # 현재 위젯 닫기

if __name__ == '__main__':
    app = QApplication(sys.argv)  # QApplication 생성
    ex = App()  # App 인스턴스 생성
    ex.show()  # 위젯 표시
    sys.exit(app.exec_())  # 이벤트 루프 실행
