from imutils.video import VideoStream  # Para manejar el flujo de video de la cámara
import face_recognition  
import imutils  # Utilidades para manipulación de imágenes
import pickle  # Para cargar y guardar datos serializados
import time
import cv2
import os
import tkinter as tk # Interfaz gráfica de usuario (GUI)
from tkinter import filedialog, messagebox

# Definir las rutas
encodings_path = r"C:\deep_learning\trabajo_final\output\encodingscnn.pickle"
output_video_path = r"C:\deep_learning\trabajo_final\output\output_video.avi"
videos_path = r"C:\deep_learning\trabajo_final\videos"
detection_method = "cnn"  # 'hog' para rapidez o 'cnn' para mayor precisión


# Función principal de la GUI
def start_gui():
    #Esta función crea una ventana gráfica para seleccionar entre usar la cámara o abrir un archivo de video.
    def use_camera():
        #Cierra la ventana GUI y llama a la función para procesar la cámara.
        root.destroy()
        start_video(0, is_video=False)

    def select_video():
        #Abre un cuadro de diálogo para seleccionar un archivo de video y llama a la función para procesarlo.
        video_file = filedialog.askopenfilename(
            title="Seleccionar Archivo de Video",
            initialdir=videos_path,
            filetypes=(("Archivos de Video", "*.mp4 *.avi *.mov"), ("Todos los Archivos", "*.*"))
        )
        if video_file:
            root.destroy()
            start_video(video_file, is_video=True)

    root = tk.Tk()
    root.title("Reconocimiento Facial")
    root.geometry("300x200")
    root.resizable(False, False)

    tk.Label(root, text="Selecciona una opción:", font=("Arial", 14)).pack(pady=20)

    tk.Button(root, text="Usar Cámara", font=("Arial", 12), command=use_camera).pack(pady=10)
    tk.Button(root, text="Abrir archivo de video", font=("Arial", 12), command=select_video).pack(pady=10)

    root.mainloop()


# Función para iniciar el procesamiento de video
def start_video(input_source, is_video):
    print("[INFO] Cargando codificaciones...")
    with open(encodings_path, "rb") as f:
        data = pickle.load(f)

    if is_video:
        print(f"[INFO] Abriendo archivo de video: {input_source}")
        vs = cv2.VideoCapture(input_source)
    else:
        print(f"[INFO] Iniciando flujo de video desde la cámara predeterminada.")
        vs = VideoStream(src=input_source).start()
        time.sleep(2.0)

    writer = None

    while True:
        frame = vs.read() if not is_video else vs.read()[1]
        if frame is None:
            break
        
        # cambiar el tamaño del cuadro para acelerar el procesamiento
        rgb = imutils.resize(frame, width=750)
        r = frame.shape[1] / float(rgb.shape[1])
        
        # convertir el cuadro de BGR a RGB (dlib requiere imágenes RGB)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        # detectar las ubicaciones de los rostros y computar las codificaciones faciales
        boxes = face_recognition.face_locations(rgb, model=detection_method)
        encodings = face_recognition.face_encodings(rgb, boxes)
        names = []

        # bucle sobre las codificaciones faciales
        for encoding in encodings:
            # intentar coincidir cada rostro en el cuadro con nuestras codificaciones conocidas
            matches = face_recognition.compare_faces(data["encodings"], encoding, tolerance=0.5)
            name = "Desconocido"
            if True in matches:
                # encontrar los índices de todas las coincidencias y contar las ocurrencias
                matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                counts = {}
                for i in matchedIdxs:
                    name = data["names"][i]
                    counts[name] = counts.get(name, 0) + 1
                # determinar el nombre con más coincidencias
                name = max(counts, key=counts.get)
            names.append(name)

        # bucle sobre los rostros reconocidos y dibujar los cuadros y nombres
        for ((top, right, bottom, left), name) in zip(boxes, names):
            # ajustar las coordenadas de acuerdo al cambio de tamaño
            top = int(top * r)
            right = int(right * r)
            bottom = int(bottom * r)
            left = int(left * r)
            # dibujar el cuadro y el nombre en el cuadro original
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            y = top - 15 if top - 15 > 15 else top + 15
            cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, (0, 255, 0), 2)

        # si se especificó un archivo de salida, inicializar el escritor
        if writer is None and output_video_path is not None:
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(output_video_path, fourcc, 20,
                                     (frame.shape[1], frame.shape[0]), True)
        # si el escritor está inicializado, escribir el cuadro en el archivo
        if writer is not None:
            writer.write(frame)

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if cv2.getWindowProperty("Frame", cv2.WND_PROP_VISIBLE) < 1:
            print("[INFO] Ventana cerrada. Finalizando...")
            break

    cv2.destroyAllWindows()
    if not is_video:
        vs.stop()
    else:
        vs.release()
    if writer is not None:
        writer.release()


# Iniciar GUI
if __name__ == "__main__":
    start_gui()
