import mediapipe as mp
import cv2
import numpy as np
import time
import psutil

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

totaltime = 0

# 入力動画ファイルのパス
input_video_path = "./samplevideos/taiso.mp4"
# 出力動画ファイルのパス
output_video_path = "./outputvideos/taisopose.mp4"

# 入力動画のキャプチャ
cap = cv2.VideoCapture(input_video_path)

# 動画の幅、高さ、フレームレートを取得
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_rate = cap.get(cv2.CAP_PROP_FPS)

# 出力動画の設定
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (frame_width, frame_height))

with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      break

    # 処理開始時間を記録
    start_time = time.time()

    # フレームを処理不可に設定
    image.flags.writeable = False
    # BGRからRGBに変換
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # 骨格検出を実行
    results = pose.process(image)

    # フレームを処理可能に設定
    image.flags.writeable = True
    # RGBからBGRに変換
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # 骨格ランドマークを描画
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

    # フレームを出力動画に書き込む
    out.write(image)

    # 画面に表示（オプション）
    cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))

    # 処理時間を計算
    processing_time = time.time() - start_time
    totaltime = totaltime + processing_time
    # メモリ使用量を取得
    process = psutil.Process()
    memory_usage = process.memory_info().rss / (1024 * 1024)  # メモリ使用量（MB）

    # 処理時間とメモリ使用量を出力
    print(f" Memory Usage: {memory_usage:.2f} MB")

    # 'ESC'キーが押されたらループを終了
    if cv2.waitKey(5) & 0xFF == 27:
      break

# 処理時間を表示
print(f" total time: {totaltime:.2f} s")

# リソースを解放
cap.release()
out.release()
cv2.destroyAllWindows()
