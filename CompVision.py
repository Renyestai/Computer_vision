import tkinter as tk
from tkinter import filedialog, messagebox, Canvas, ttk
import cv2
from PIL import Image, ImageTk
from ultralytics import YOLO
import os
import threading
import queue
import time
import torch
import ttkbootstrap as ttkb

speed_factor = 1.0  # Начальная скорость воспроизведения
if torch.cuda.is_available():
    device = 'cuda'
    print("CUDA доступен. Используется GPU.")
else:
    device = 'cpu'
    print("CUDA недоступен. Используется CPU.")

# Загружаем модель YOLO
model = YOLO('weights/best.pt')
model.to(device)  # Перенос модели на устройство
print(f"YOLO работает на: {model.device}")

# Очередь для передачи кадров между потоками
frame_queue = queue.Queue(maxsize=1)
stop_flag = threading.Event()
pause_flag = threading.Event()
restart_flag = threading.Event()

current_video_path = None  # Хранит путь к текущему видео

def upload_video():
    global current_video_path 
    default_video_folder = os.path.join(os.environ['USERPROFILE'], 'Videos')
    filepath = filedialog.askopenfilename(
        initialdir=default_video_folder,
        filetypes=[("Video Files", "*.mp4;*.avi;*.mov")]
    )
    if filepath:
        stop_current_video()
        # Проверяем, существует ли окно и текстовая область
        if hasattr(open_logs_window, 'text_area'):
            open_logs_window.text_area.delete(1.0, tk.END)

        frame_queue.queue.clear()
        stop_flag.clear()
        pause_flag.clear()
        restart_flag.clear()
        #progress_bar["value"] = 0  # Сброс прогресса
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
        open_logs_window.text_area.delete(1.0, tk.END)  # Очищаем логи
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
    global speed_factor  # Делаем переменную глобальной
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_delay = (1 / fps) / speed_factor  # Задержка с учётом скорости
    current_frame = 0
    analyze_every_nth_frame = 2  # Анализировать каждый 2-й кадр для снижения нагрузки

    while not stop_flag.is_set():
        if pause_flag.is_set():  # Пауза
            time.sleep(0.1)
            continue

        ret, frame = cap.read()
        if not ret:  # Если видео закончилось
            break

        current_frame += 1

        # Конвертируем кадр в RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if enabled.get() and (current_frame % analyze_every_nth_frame == 0):  # Если анализ включён
            results = model(frame_rgb, conf=0.5)  # Анализируем кадры с интервалом
            annotated_frame = results[0].plot()
            update_logs(results)
        else:
            annotated_frame = frame_rgb

        annotated_frame = cv2.resize(annotated_frame, (800, 600))

        if not frame_queue.full():
            frame_queue.put(annotated_frame)

        # Обновление прогресс-бара
        progress = (current_frame / total_frames) * 100
        #progress_bar["value"] = progress

        time.sleep(frame_delay)  # Задержка с учётом скорости

    cap.release()


def update_frame():
    """Обновляет кадры на Canvas в главном потоке."""
    if not frame_queue.empty():
        frame = frame_queue.get()
        
        # Получаем размеры изображения
        video_height, video_width = frame.shape[0], frame.shape[1]

        root.update()  # Обновляем окно, чтобы учесть размеры всех элементов

        # Получаем высоту окна
        window_height = root.winfo_height()

        # Высота верхней панели (frame_top), которая занимает часть окна
        top_panel_height = frame_top.winfo_height()  

        # Высота доступного пространства для канваса
        canvas_height = window_height - top_panel_height - 100

       # Вычисляем масштабный коэффициент
        scale_factor = canvas_height / video_height

        # Масштабируем размеры видео
        new_width = int(video_width * scale_factor)
        new_height = canvas_height

        # Масштабируем кадр
        frame_resized = cv2.resize(frame, (new_width, new_height))
        img = ImageTk.PhotoImage(Image.fromarray(frame_resized))
        canvas.delete("all")
        canvas.create_image((canvas.winfo_width() - new_width) // 2, (canvas.winfo_height() - new_height) // 2, anchor=tk.NW, image=img)
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
        open_logs_window.text_area.insert(tk.END, logs)
        open_logs_window.text_area.see(tk.END)
    
def toggle_speed():
    """Переключение скорости воспроизведения."""
    global speed_factor
    if speed_factor == 1.0:
        speed_factor = 2.0  # Ускоряем в 2 раза
        btn_speed.config(text="Скорость x2")
    else:
        speed_factor = 1.0  # Возвращаем нормальную скорость
        btn_speed.config(text="Скорость x1")  # Меняем текст кнопки

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

def open_logs_window():
    # Проверяем, если окно уже открыто, то не открываем его снова
    if not hasattr(open_logs_window, "window") or open_logs_window.window.winfo_exists() == 0:
        open_logs_window.window = tk.Toplevel(root)
        open_logs_window.window.geometry("400x720")
        open_logs_window.window.title("Логи")
        
        open_logs_window.text_area = tk.Text(open_logs_window.window, wrap=tk.WORD, font=("Consolas", 10))
        open_logs_window.text_area.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        btn = ttk.Button(open_logs_window.window, text="Закрыть окно", command=close_logs_window, bootstyle="info-outline")
        btn.pack(side=tk.BOTTOM, pady=(5, 20))

        # Обработчик закрытия окна через X
        open_logs_window.window.protocol("WM_DELETE_WINDOW", on_close_logs_window)

        # Скрываем окно сразу после его создания
        open_logs_window.window.withdraw()

# Добавление обработчика для закрытия окна через X
def on_close_logs_window():
    close_logs_window()  

def close_logs_window():
    # Скрыть окно, а не закрывать его
    if hasattr(open_logs_window, "window"):
        open_logs_window.window.withdraw()

def toggle_logs_window():
    # Если окно уже открыто, показываем или скрываем его
    if hasattr(open_logs_window, "window"):
        if open_logs_window.window.winfo_ismapped(): 
            open_logs_window.window.withdraw()  
        else:
            open_logs_window.window.deiconify()  

# Создание главного окна
root = ttkb.Window(themename="solar")
root.title("Обнаружение повреждений на дороге")
root.geometry("940x720")
#root.resizable(width=False, height=False)

# Верхняя панель
frame_top = ttkb.Frame(root, padding=10)
frame_top.pack(fill=tk.X)

btn_upload = ttkb.Button(frame_top, text="Загрузить видео", command=upload_video, bootstyle="success-outline")
btn_upload.pack(side=tk.LEFT, padx=20, pady=(15, 0))

btn_restart = ttkb.Button(frame_top, text="Запустить сначала", command=restart_video, bootstyle="warning-outline")
btn_restart.pack(side=tk.LEFT, padx=15, pady=(15, 0))

btn_pause_resume = ttkb.Button(frame_top, text="Пауза", width=12, command=toggle_pause, bootstyle="primary-outline")
btn_pause_resume.pack(side=tk.LEFT, padx=15, pady=(15, 0))

btn_speed = ttkb.Button(frame_top, text="Скорость x2", command=toggle_speed, bootstyle="info-outline")
btn_speed.pack(side=tk.LEFT, padx=15, pady=(15, 0)) 

btn_logs = ttk.Button(frame_top, text="Открыть логи", command=toggle_logs_window, bootstyle="info-outline")
btn_logs.pack(side=tk.LEFT, padx=15, pady=(15, 0))

enabled = tk.IntVar()
enabled_checkbutton = ttkb.Checkbutton(frame_top, text="Включить поиск повреждений", variable=enabled, bootstyle="info")
enabled_checkbutton.pack(side=tk.LEFT, padx=20, pady=(15, 0))

# Canvas для видео
canvas = Canvas(root, background="black")
canvas.pack(pady=30, padx=30, expand=True, fill=tk.BOTH)

# Прогресс-бар я думаю, он не нужен
#progress_bar = ttkb.Progressbar(root, length=500, mode="determinate") 
#progress_bar.pack(side=tk.RIGHT, padx=20, pady=(60, 0))
#progress_bar.place(x=600, y=660)

root.protocol("WM_DELETE_WINDOW", on_close)

open_logs_window()
root.mainloop()
