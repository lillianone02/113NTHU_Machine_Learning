import cv2
import dlib
import os

# 初始化 Dlib 的人臉檢測器（HOG）和68點特徵標註模型
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def detect_face_and_landmarks(frame):
    """
    使用 Dlib 檢測人臉並標註 68 點特徵。
    Args:
        frame (numpy.ndarray): 輸入影像 (RGB 或 BGR 格式皆可，不過此處使用 BGR 格式)。
    Returns:
        frame (numpy.ndarray): 標註了人臉特徵點的影像 (與輸入大小相同)。
    """
    # 確保為 BGR -> 灰階
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray_frame)  # 偵測人臉

    for face in faces:
        # 繪製人臉框
        x, y, x1, y1 = face.left(), face.top(), face.right(), face.bottom()
        # cv2.rectangle(frame, (x, y), (x1, y1), (255, 0, 0), 2)藍色方框

        # 獲取 68 點特徵
        landmarks = predictor(gray_frame, face)
        for n in range(68):
            lx = landmarks.part(n).x
            ly = landmarks.part(n).y
            cv2.circle(frame, (lx, ly), 2, (0, 255, 0), -1)

    return frame

def process_single_image(input_path, output_path):
    """
    處理單張圖片，檢測人臉並標記特徵點，保存到輸出路徑。
    Args:
        input_path (str): 輸入圖片的檔案路徑。
        output_path (str): 輸出圖片的檔案路徑。
    """
    # 確保輸入圖片存在
    if not os.path.exists(input_path):
        print(f"輸入檔案不存在：{input_path}")
        return

    # 讀取圖片
    image = cv2.imread(input_path)
    if image is None:
        print(f"無法讀取圖片：{input_path}")
        return

    # 處理圖片
    processed_image = detect_face_and_landmarks(image)

    # 保存處理後的圖片
    cv2.imwrite(output_path, processed_image)
    print(f"處理完成，結果保存至：{output_path}")

if __name__ == "__main__":
    # 替換成你自己的路徑
    input_path = "your_path_to_image"  # 單張圖片的路徑
    output_path = "your_path_to_image"  # 處理後圖片的保存路徑

    process_single_image(input_path, output_path)
