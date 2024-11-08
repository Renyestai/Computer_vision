import tkinter as tk
from tkinter import filedialog, messagebox, Canvas,ttk
import cv2
from PIL import Image, ImageTk
from ultralytics import YOLO
from threading import Thread
import socket
import subprocess
# Загружаем модель YOLO
model = YOLO('weights/best.pt')  # Убедитесь, что путь указан правильно



def upload_video():
    filepath = filedialog.askopenfilename(
        filetypes=[("Video Files", "*.mp4;*.avi;*.mov")]
    )
    if filepath:
        
        text_area.delete(1.0, tk.END)
        play_video(filepath)



def play_video(video_path):
    global cap  #  Определяем  cap  как  глобальную  переменную,  чтобы  она  была  доступна  внутри  функции  update_frame

    # Захват видео с помощью OpenCV
    cap = cv2.VideoCapture(video_path)

    def update_frame():
        global cap  
        ret, frame = cap.read()
        results = []

        if ret:
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if enabled.get():
                
                def run_yolo():
                    nonlocal results
                    results = model(frame_rgb) 
                    update_logs(results)  

                yolo_thread = Thread(target=run_yolo)
                yolo_thread.start()

                
                yolo_thread.join() 

                
                annotated_frame = results[0].plot()
                annotated_frame = cv2.resize(annotated_frame, (800, 600))
            else:
                annotated_frame = cv2.resize(frame_rgb, (800, 600))

            img = ImageTk.PhotoImage(Image.fromarray(annotated_frame))
            canvas.create_image(0, 0, anchor=tk.NW, image=img)
            canvas.image = img 

            # Продолжаем обновлять кадры
            root.after(20, update_frame)
        else:
            cap.release()
            messagebox.showinfo("Видео завершено", "Воспроизведение видео завершено")

    # Ожидание полной инициализации окна перед началом обновления кадра
    root.after(200, update_frame)



def detect_damage():
    
    messagebox.showinfo("Результат", "Анализ завершен!")

def update_logs(results):
    
    detections = results[0].boxes
    logs = ""

   
    for detection in detections:
        if detection.conf >= 0.7:  
            class_id = int(detection.cls)  
            class_names = ["Предположительно светофоры", "Поврежденные светофоры", "Ямы?", "Знак_Поврежденный", "Знак_Нормальный"]
            confidence = detection.conf.item()  
            logs += f"Обнаружен: класс {class_names[class_id]} с уверенностью {confidence:.2f}\n"

    # Обновляем виджет Text с логами
    if logs:
        text_area.insert(tk.END, logs)
        text_area.see(tk.END)  
    
def on_button_click():
    rtsp_url = entry_rtsp.get()  
 
    if rtsp_url:
    
        def play_rtsp_stream():
            cap = cv2.VideoCapture(rtsp_url)
            if cap.isOpened():
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    cv2.imshow("RTSP Stream", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                cap.release()
                cv2.destroyAllWindows()
            else:
                print("Ошибка открытия потока.")

        thread = Thread(target=play_rtsp_stream)
        thread.start()
   

# Создание главного окна
root = tk.Tk()
root.title("Обнаружение повреждений на дороге")
root.geometry("1280x720")

# для загрузки видео
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
canvas = Canvas(root, width=800, height=600, state="disabled")
canvas.place(relx=1.0, rely=0.0, anchor=tk.NE)  

# отображения логов
text_area = tk.Text(root, height=20, width=40)
text_area.place(x=20, y=120)  

button_rtsp = tk.Button(root, text="Найти камеру", command=on_button_click)
button_rtsp.place(x=200, y=20)

entry_rtsp = ttk.Entry(root, width=30)
entry_rtsp.pack()
entry_rtsp.place(x=200, y=60)

# Запуск интерфейса
root.mainloop()