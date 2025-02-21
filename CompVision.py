
import tkinter as tk
from tkinter import filedialog, messagebox, Canvas, ttk
import cv2
from PIL import Image, ImageTk
from ultralytics import YOLO
import threading
import queue
import time
import ttkbootstrap as ttkb  # Библиотека для стилизации
# Переменная для управления скоростью воспроизведения
speed_factor = 1.0  # Начальная скорость воспроизведения (1x)

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
        stop_current_video()
        text_area.delete(1.0, tk.END)
        frame_queue.queue.clear()
        stop_flag.clear()
        pause_flag.clear()
        restart_flag.clear()
        progress_bar["value"] = 0  # Сброс прогресса
        btn_pause_resume.config(text="Пауза")
        current_video_path = filepath
        threading.Thread(target=process_video, args=(filepath,), daemon=True).start()
        update_frame()

def stop_current_video():
    """Останавливает текущее видео, гарантируя завершение процесса."""
    stop_flag.set()  # Устанавливаем флаг остановки
    while not frame_queue.empty():
        try:
            frame_queue.get_nowait()  # Очищаем очередь кадров
        except queue.Empty:
            break
    time.sleep(0.1)  # Даем немного времени для завершения текущего потока

def restart_video():
    """Полностью перезапускает видео с очисткой всех процессов."""
    global current_video_path
    if current_video_path:
        stop_current_video()  # Останавливаем текущий поток
        text_area.delete(1.0, tk.END)  # Очищаем логи
        stop_flag.clear()  # Сбрасываем флаг остановки
        pause_flag.clear()  # Снимаем паузу
        restart_flag.clear()  # Убираем флаг перезапуска
        btn_pause_resume.config(text="Пауза")  # Сбрасываем текст кнопки паузы
        # Запускаем новое воспроизведение видео
        threading.Thread(target=process_video, args=(current_video_path,), daemon=True).start()
        update_frame()
    else:
        messagebox.showwarning("Ошибка", "Видео не загружено. Загрузите видео для перезапуска.")

def process_video(video_path):
    """Обработка видео с учётом скорости и флагов."""
    global speed_factor  # Используем глобальную переменную для управления скоростью
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_delay = 1 / fps  # Базовая задержка без учёта скорости
    current_frame = 0
    analyze_every_nth_frame = 2  # Анализировать каждый 2-й кадр для снижения нагрузки

    while not stop_flag.is_set():
        if pause_flag.is_set():  # Если пауза включена
            time.sleep(0.1)
            continue

        ret, frame = cap.read()
        if not ret:  # Если видео закончилось
            break

        current_frame += 1

        # Конвертируем кадр в RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if enabled.get() and (current_frame % analyze_every_nth_frame == 0):  # Если анализ включён
            results = model(frame_rgb, conf=0.5)  # YOLO анализирует кадры
            annotated_frame = results[0].plot()
            update_logs(results)
        else:
            annotated_frame = frame_rgb

        annotated_frame = cv2.resize(annotated_frame, (800, 600))

        if not frame_queue.full():
            frame_queue.put(annotated_frame)

        # Рассчитываем задержку с учётом текущей скорости
        adjusted_delay = frame_delay / speed_factor
        time.sleep(adjusted_delay)  # Задержка между кадрами с учётом скорости

        # Обновление прогресс-бара
        progress = (current_frame / total_frames) * 100
        progress_bar["value"] = progress

    cap.release()



def update_frame():
    """Обновляет кадры на Canvas в главном потоке."""
    if not frame_queue.empty():
        frame = frame_queue.get()
        img = ImageTk.PhotoImage(Image.fromarray(frame))
        canvas.create_image(0, 0, anchor=tk.NW, image=img)
        canvas.image = img

    if not stop_flag.is_set():
        root.after(20, update_frame)
    else:
        canvas.delete("all")  #

def update_logs(results):
    detections = results[0].boxes
    logs = ""

    class_names = model.names

    for detection in detections:
        class_id = int(detection.cls)
        class_name = class_names[class_id]
        confidence = detection.conf.item()
        logs += f"Обнаружено: класс {class_name} с уверенностью {confidence:.2f}\n"

    if logs:
        text_area.insert(tk.END, logs)
        text_area.see(tk.END)
    
def toggle_speed():
    """Переключение скорости воспроизведения."""
    global speed_factor
    if speed_factor == 1.0:
        speed_factor = 2.0  # Ускоряем в 2 раза
        btn_speed.config(text="Скорость x2")  # Меняем текст кнопки
    else:
        speed_factor = 1.0  # Возвращаем нормальную скорость
        btn_speed.config(text="Скорость x1")

def toggle_pause():
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
root = ttkb.Window(themename="solar")
root.title("Обнаружение повреждений на дороге")
root.geometry("1280x720")
root.resizable(width=False, height=False)

# Верхняя панель
# Верхняя панель
# Верхняя панель
frame_top = ttkb.Frame(root, padding=10)
frame_top.pack(fill=tk.X)

lbl_title = ttkb.Label(frame_top, text="Система анализа дорожных повреждений", font=("Arial", 16), bootstyle="dark")
lbl_title.pack(side=tk.LEFT, padx=10)

btn_upload = ttkb.Button(frame_top, text="Загрузить видео", command=upload_video, bootstyle="success-outline")
btn_upload.pack(side=tk.LEFT, padx=10)

btn_restart = ttkb.Button(frame_top, text="Запустить сначала", command=restart_video, bootstyle="warning-outline")
btn_restart.pack(side=tk.LEFT, padx=10)

btn_pause_resume = ttkb.Button(frame_top, text="Пауза", command=toggle_pause, bootstyle="primary-outline")
btn_pause_resume.pack(side=tk.LEFT, padx=10)

btn_speed = ttkb.Button(frame_top, text="Скорость x2", command=toggle_speed, bootstyle="info-outline")
btn_speed.pack(side=tk.LEFT, padx=10)  # Рядом с кнопкой "Пауза"

enabled = tk.IntVar()
enabled_checkbutton = ttkb.Checkbutton(frame_top, text="Включить поиск повреждений", variable=enabled, bootstyle="info")
enabled_checkbutton.pack(side=tk.LEFT, padx=20)

# Прогресс-бар
progress_bar = ttkb.Progressbar(root, length=500, mode="determinate")
progress_bar.pack(pady=10)

# Canvas для видео
canvas = Canvas(root, width=800, height=600, background="black")
canvas.pack(side=tk.RIGHT, padx=20, pady=20)

# Логи
text_area_frame = ttkb.Labelframe(root, text="Логи", padding=10, bootstyle="dark")
text_area_frame.pack(side=tk.LEFT, fill=tk.Y, padx=20, pady=20)

text_area = tk.Text(text_area_frame, height=30, width=50, wrap=tk.WORD, font=("Consolas", 10))
text_area.pack()

# Информация о модели
info_frame = ttkb.Labelframe(root, text="Информация о модели", padding=10, bootstyle="dark")
info_frame.pack(side=tk.LEFT, fill=tk.X, padx=20, pady=20)
model_info = f"YOLO: {model.__class__.__name__}\nУстройство: {model.device}"
ttkb.Label(info_frame, text=model_info, bootstyle="secondary").pack()

root.protocol("WM_DELETE_WINDOW", on_close)
root.mainloop()
