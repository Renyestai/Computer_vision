import sqlite3
import os
import cv2


#База_данных
def create_db():
    conn= sqlite3.connect("detections.db")
    cursor =conn.cursor()
    cursor.execute("""CREATE TABLE IF NOT EXISTS detections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    object_name TEXT,
                    confidence REAL,
                    latitude REAL,
                    longitude REAL,
                    image_path TEXT)""" )
    conn.commit()
    conn.close()
#Сохранка для обнаруженного неронкой
def save_detection(object_name,confidence,latitude,longitude,frame):
    conn= sqlite3.connect("detections.db")
    cursor =conn.cursor()

    #Путь храниения 
    image_folder ="detection_img"
    os.makedirs(image_folder,exist_ok=True)

    image_name=f"{object_name}_{latitude}_{longitude}.jpg"
    image_path=os.path.join(image_folder,image_name)

    cv2.imwrite(image_path,frame)


    #В базу данных
    cursor.execute("INSERT INTO detections (object_name,confidence,latitude,longitude,image_path) VALUES(?,?,?,?,?)",
                  (object_name,confidence,latitude,longitude,image_path))
    conn.commit()
    conn.close()
    print(f"Данные сохранены: {object_name} ({confidence*100:.2f}%) → {image_path}")


#функция для получения всего из бд
def get_detections():
    conn= sqlite3.connect("detections.db")
    cursor =conn.cursor()
    cursor.execute("SELECT * FROM detections")
    results= cursor.fetchall()
    conn.close()
    return results
#вызов при запуске модуля 
create_db()