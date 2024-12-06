# Importar los paquetes necesarios
import face_recognition
import pickle
import cv2
import os

# Definir las rutas directamente en el script
encodings_path = r"C:\deep_learning\trabajo_final\output\encodings.pickle"
# image_path = r"C:\deep_learning\trabajo_final\dataset\jeff_bridges\005.jpg"
image_path = r"C:\deep_learning\trabajo_final\examples\example_02.png"
detection_method = "hog"  # cnn, cambia a "hog" si prefieres mayor velocidad

# Cargar las codificaciones conocidas
print("[INFO] Cargando codificaciones...")
with open(encodings_path, "rb") as f:
    data = pickle.load(f)

# Cargar la imagen de entrada y convertirla de BGR a RGB
image = cv2.imread(image_path)
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Detectar las ubicaciones de los rostros en la imagen de entrada
print("[INFO] Reconociendo rostros...")
boxes = face_recognition.face_locations(rgb, model=detection_method)
encodings = face_recognition.face_encodings(rgb, boxes)

# Inicializar la lista de nombres para cada rostro detectado
names = []

# Bucle sobre las codificaciones faciales
for encoding in encodings:
    # Intentar coincidir cada rostro en la imagen de entrada con nuestras codificaciones conocidas
    matches = face_recognition.compare_faces(data["encodings"], encoding)
    name = "Desconocido"

    # Verificar si encontramos una coincidencia
    if True in matches:
        # Encontrar los índices de todas las coincidencias y contar cuántas veces aparece cada una
        matchedIdxs = [i for (i, b) in enumerate(matches) if b]
        counts = {}
        for i in matchedIdxs:
            name = data["names"][i]
            counts[name] = counts.get(name, 0) + 1
        # Determinar el nombre con más coincidencias
        name = max(counts, key=counts.get)

    # Actualizar la lista de nombres
    names.append(name)

# Mostrar los resultados
for ((top, right, bottom, left), name) in zip(boxes, names):
    # Dibujar el nombre del rostro predicho en la imagen
    cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
    y = top - 15 if top - 15 > 15 else top + 15
    cv2.putText(image, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
                0.75, (0, 255, 0), 2)

# Mostrar la imagen de salida
cv2.imshow("Image", image)
cv2.waitKey(0)
