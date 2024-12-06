# importar los paquetes necesarios
from imutils.video import VideoStream
import face_recognition
import imutils
import pickle
import time
import cv2

# Definir las rutas y configuraciones directamente en el script
encodings_path = r"C:\deep_learning\trabajo_final\output\encodings.pickle"
output_video_path = r"C:\deep_learning\trabajo_final\output\output_video.avi"
display = 1  # Establece a 0 si no deseas mostrar el video en pantalla
detection_method = "hog"  #hog  O "cnn" para mayor precisión

# cargar las codificaciones de rostros conocidas
print("[INFO] cargando codificaciones...")
with open(encodings_path, "rb") as f:
    data = pickle.load(f)

# inicializar el flujo de video y el puntero al archivo de video de salida
print("[INFO] iniciando flujo de video...")
vs = VideoStream(src=0).start()
writer = None
time.sleep(2.0)

# bucle sobre los cuadros del flujo de video
while True:
    # tomar el cuadro del flujo de video
    frame = vs.read()
    
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

    # mostrar el cuadro en pantalla si está habilitado
    if display > 0:
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        # si se presiona 'q', salir del bucle
        if key == ord("q"):
            break

# realizar una limpieza
cv2.destroyAllWindows()
vs.stop()
# liberar el escritor si se utilizó
if writer is not None:
    writer.release()