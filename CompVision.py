import tkinter as tk
from tkinter import filedialog, messagebox, Canvas
import cv2
from PIL import Image, ImageTk
from ultralytics import YOLO

# Загружаем модель YOLO
model = YOLO('weights/best.pt')  # Убедитесь, что путь указан правильно

def upload_video():
    filepath = filedialog.askopenfilename(
        filetypes=[("Video Files", "*.mp4;*.avi;*.mov")]
    )
    if filepath:
        # Очистка текстовой области при загрузке нового видео
        text_area.delete(1.0, tk.END)
        play_video(filepath)

def play_video(video_path):
    # Захват видео с помощью OpenCV
    cap = cv2.VideoCapture(video_path)

    def update_frame():
        ret, frame = cap.read()
        results = []

        if ret:
            # Конвертируем кадр в формат RGB для отображения в Tkinter
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if enabled.get():
                # Запуск модели YOLO для анализа текущего кадра
                results = model(frame_rgb)

                # Получаем кадр с нанесенными детекциями
                annotated_frame = results[0].plot()
                update_logs(results)  # Обновление логов

                # Изменяем размер кадра для отображения в Tkinter
                annotated_frame = cv2.resize(annotated_frame, (800, 600))
            else:
                annotated_frame = cv2.resize(frame_rgb,  (800, 600))

            img = ImageTk.PhotoImage(Image.fromarray(annotated_frame))


            # Обновляем Canvas новым изображением
            canvas.create_image(0, 0, anchor=tk.NW, image=img)
            canvas.image = img  # Обновляем изображение в зоне Canvas

            # Продолжаем обновлять кадры
            root.after(20, update_frame)
        else:
            cap.release()
            messagebox.showinfo("Видео завершено", "Воспроизведение видео завершено")

    # Ожидание полной инициализации окна перед началом обновления кадра
    root.after(200, update_frame)

def detect_damage():
    # Функция анализа изображения на предмет повреждений
    messagebox.showinfo("Результат", "Анализ завершен!")

def update_logs(results):
    # Получаем найденные объекты
    detections = results[0].boxes
    logs = ""

    # Собираем информацию о найденных повреждениях
    for detection in detections:
        if detection.conf >= 0.2:  # Пример порога уверенности
            class_id = int(detection.cls)  # ID класса
            confidence = detection.conf.item()  # Уверенность
            logs += f"Обнаружено: класс {class_id} с уверенностью {confidence:.2f}\n"

    # Обновляем виджет Text с логами
    if logs:
        text_area.insert(tk.END, logs)
        text_area.see(tk.END)  # Прокручиваем текст вниз

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
enabled_checkbutton = tk.Checkbutton(text="Включить поиск повреждений", variable=enabled)
enabled_checkbutton.pack(padx=10, pady=100, anchor=tk.NW)

# Создаем Canvas для отображения видео
canvas = Canvas(root, width=800, height=600)
canvas.place(relx=1.0, rely=0.0, anchor=tk.NE)  # Располагаем в окне с отступом

# Создаем виджет Text для отображения логов
text_area = tk.Text(root, height=20, width=40)
text_area.place(x=20, y=120)  # Позиционируем виджет

# Запуск интерфейса
root.mainloop()