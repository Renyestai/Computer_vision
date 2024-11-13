import tkinter as tk
from tkinter import filedialog, messagebox, Canvas
import cv2
from PIL import Image, ImageTk
from ultralytics import YOLO
import threading
import queue
import time
import torch

print("Поддержка CUDA в PyTorch:", torch.cuda.is_available())

# Установка устройства: GPU (cuda), если доступен, иначе CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLO('weights/best.pt').to(device)

# Очередь для передачи кадров между потоками
frame_queue = queue.Queue(maxsize=1)

# Флаг для остановки потоков и переключения состояния
stop_flag = threading.Event()
recognition_mode = threading.Event()  # Состояние распознавания

def upload_video():
    filepath = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4;*.avi;*.mov")])
    if filepath:
        text_area.delete(1.0, tk.END)
        stop_flag.clear()
        recognition_mode.clear()  # По умолчанию без распознавания
        threading.Thread(target=process_video, args=(filepath,), daemon=True).start()
        update_frame()  # Запуск обновления кадров интерфейса

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    while not stop_flag.is_set():
        ret, frame = cap.read()
        if not ret:
            break

        # Конвертируем кадр в RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Проверка состояния распознавания
        if recognition_mode.is_set():
            # Включен режим распознавания — добавляем результаты модели
            results = model(frame_rgb)
            annotated_frame = results[0].plot()
            
            # Обновляем логи
            update_logs(results)
            print("YOLO выполняется на устройстве:", next(model.parameters()).device)
        else:
            # Режим без распознавания — обычный кадр
            annotated_frame = frame_rgb

            # Добавляем небольшую задержку для стабилизации частоты кадров
            time.sleep(0.03)

        # Изменяем размер кадра
        annotated_frame = cv2.resize(annotated_frame, (800, 600))

        # Отправляем кадр в очередь для отображения
        if not frame_queue.full():
            frame_queue.put(annotated_frame)

        # Включаем небольшую задержку, чтобы снизить нагрузку
        time.sleep(0.01)
    
    cap.release()

def update_frame():
    # Проверка, есть ли кадры в очереди, и обновление Canvas
    if not frame_queue.empty():
        frame = frame_queue.get()
        img = ImageTk.PhotoImage(Image.fromarray(frame))
        
        # Обновляем Canvas новым изображением
        canvas.create_image(0, 0, anchor=tk.NW, image=img)
        canvas.image = img

    # Продолжаем обновлять кадры каждые 10 миллисекунд
    if not stop_flag.is_set():
        root.after(20, update_frame)
    else:
        messagebox.showinfo("Видео завершено", "Воспроизведение видео завершено")

def toggle_recognition():
    # Переключение состояния между простым выводом и распознаванием
    if enabled.get():
        recognition_mode.set()  # Включаем распознавание
    else:
        recognition_mode.clear()  # Отключаем распознавание

def detect_damage():
    messagebox.showinfo("Результат", "Анализ завершен!")

def update_logs(results):
    detections = results[0].boxes
    logs = ""
    
    for detection in detections:
        if detection.conf >= 0.2:
            class_id = int(detection.cls)
            confidence = detection.conf.item()
            logs += f"Обнаружено: класс {class_id} с уверенностью {confidence:.2f}\n"

    if logs:
        text_area.insert(tk.END, logs)
        text_area.see(tk.END)

def on_close():
    stop_flag.set()  # Устанавливаем флаг остановки
    root.destroy()   # Закрываем окно

# Создание главного окна
root = tk.Tk()
root.title("Обнаружение повреждений на дороге")
root.geometry("1280x720")

# Кнопка для загрузки видео
btn_upload = tk.Button(root, text="Загрузить видео", command=upload_video)
btn_upload.place(x=20, y=20)

# Кнопка для анализа
btn_detect = tk.Button(root, text="Обнаружить повреждения", command=detect_damage)
btn_detect.place(x=20, y=70)

# CheckBox для включения анализа
enabled = tk.IntVar()
enabled_checkbutton = tk.Checkbutton(text="Включить поиск повреждений", variable=enabled, command=toggle_recognition)
enabled_checkbutton.pack(padx=10, pady=100, anchor=tk.NW)

# Создаем Canvas для отображения видео
canvas = Canvas(root, width=800, height=600)
canvas.place(relx=1.0, rely=0.0, anchor=tk.NE)

# Создаем виджет Text для отображения логов
text_area = tk.Text(root, height=20, width=40)
text_area.place(x=20, y=120)

# Настраиваем закрытие окна с остановкой потока
root.protocol("WM_DELETE_WINDOW", on_close)

# Запуск интерфейса
root.mainloop()
