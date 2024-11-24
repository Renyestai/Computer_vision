import tkinter as tk
from tkinter import filedialog, messagebox, Canvas
import cv2
from PIL import Image, ImageTk
from ultralytics import YOLO
import threading
import queue
import time

# Загружаем модель YOLO
model = YOLO('weights/best.pt')
print(f"YOLO работает на: {model.device}")

# Очередь для передачи кадров между потоками
frame_queue = queue.Queue(maxsize=1)
stop_flag = threading.Event()
pause_flag = threading.Event()
restart_flag = threading.Event()

current_video_path = None  # Хранит путь к текущему видео

def upload_video():
    global current_video_path
    filepath = filedialog.askopenfilename(
        filetypes=[("Video Files", "*.mp4;*.avi;*.mov")]
    )
    if filepath:
        # Полная очистка перед загрузкой нового видео
        stop_current_video()
        text_area.delete(1.0, tk.END)  # Очистка логов
        frame_queue.queue.clear()  # Очистка очереди кадров
        stop_flag.clear()
        pause_flag.clear()
        restart_flag.clear()
        btn_pause_resume.config(text="Пауза")  # Сброс текста кнопки
        current_video_path = filepath
        threading.Thread(target=process_video, args=(filepath,), daemon=True).start()
        update_frame()

def stop_current_video():
    """Останавливает текущее видео."""
    stop_flag.set()  # Останавливаем текущий поток
    time.sleep(0.1)  # Даем время потоку завершиться

def restart_video():
    """Перезапускает текущее видео с начала и очищает логи."""
    global current_video_path
    if current_video_path:
        stop_current_video()
        frame_queue.queue.clear()  # Очистка очереди кадров
        text_area.delete(1.0, tk.END)  # Очищаем логи
        stop_flag.clear()
        pause_flag.clear()
        restart_flag.clear()
        btn_pause_resume.config(text="Пауза")  # Сброс текста кнопки
        threading.Thread(target=process_video, args=(current_video_path,), daemon=True).start()
        update_frame()
    else:
        messagebox.showwarning("Ошибка", "Видео не загружено. Загрузите видео для перезапуска.")

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_delay = 1 / fps

    while not stop_flag.is_set():
        if pause_flag.is_set():
            time.sleep(0.1)  # Ожидаем, пока пауза не будет снята
            continue

        if restart_flag.is_set():
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Возвращаемся к первому кадру
            restart_flag.clear()

        ret, frame = cap.read()
        if not ret:
            break

        # Конвертируем кадр в RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if enabled.get():
            results = model(frame_rgb, conf=0.5)
            annotated_frame = results[0].plot()
            update_logs(results)
        else:
            annotated_frame = frame_rgb

        # Изменяем размер кадра
        annotated_frame = cv2.resize(annotated_frame, (800, 600))

        # Отправляем кадр в очередь
        if not frame_queue.full():
            frame_queue.put(annotated_frame)

        elapsed_time = time.time() - time.time()
        delay = max(0, frame_delay - elapsed_time)
        time.sleep(delay)

    cap.release()

def update_frame():
    if not frame_queue.empty():
        frame = frame_queue.get()
        img = ImageTk.PhotoImage(Image.fromarray(frame))

        # Обновляем Canvas новым изображением
        canvas.create_image(0, 0, anchor=tk.NW, image=img)
        canvas.image = img

    if not stop_flag.is_set():
        root.after(20, update_frame)
    else:
        canvas.delete("all")  # Очистка Canvas при завершении видео

def update_logs(results):
    detections = results[0].boxes
    logs = ""

    class_names = model.names

    for detection in detections:
        class_id = int(detection.cls)
        class_name = class_names[class_id]  # Название класса
        confidence = detection.conf.item()
        logs += f"Обнаружено: класс {class_name} с уверенностью {confidence:.2f}\n"

    if logs:
        text_area.insert(tk.END, logs)
        text_area.see(tk.END)

def toggle_pause():
    """Переключает паузу и изменяет текст кнопки."""
    if pause_flag.is_set():
        pause_flag.clear()
        btn_pause_resume.config(text="Пауза")
    else:
        pause_flag.set()
        btn_pause_resume.config(text="Продолжить")

def on_close():
    stop_flag.set()
    root.destroy()

# Создание главного окна
root = tk.Tk()
root.title("Обнаружение повреждений на дороге")
root.geometry("1280x720")
root.resizable(width=False, height=False)  

# Кнопка для загрузки видео
btn_upload = tk.Button(root, text="Загрузить видео", command=upload_video)
btn_upload.place(x=20, y=40)

# Объединённая кнопка пауза/продолжить
btn_pause_resume = tk.Button(root, text="Пауза", command=toggle_pause)
btn_pause_resume.place(x=255, y=40)

# Кнопка для перезапуска видео
btn_restart = tk.Button(root, text="Запустить сначала", command=restart_video)
btn_restart.place(x=130, y=40)

# CheckBox для включения анализа
enabled = tk.IntVar()
enabled_checkbutton = tk.Checkbutton(text="Включить поиск повреждений", variable=enabled)
enabled_checkbutton.pack(padx=20, pady=90, anchor=tk.NW)

# Создаем Canvas для отображения видео
canvas = Canvas(root, width=800, height=600, background="lightgray")
canvas.place(x=450, y=40)
#canvas.place(relx=1.0, rely=0.0, anchor=tk.NE)

# Создаем виджет Text для отображения логов
text_area = tk.Text(root, height=20, width=50)
text_area.place(x=20, y=140)

# Настраиваем закрытие окна
root.protocol("WM_DELETE_WINDOW", on_close)

# Запуск интерфейса
root.mainloop()
