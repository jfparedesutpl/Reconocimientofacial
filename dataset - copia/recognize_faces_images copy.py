# Importar los paquetes necesarios
import face_recognition
import pickle
import cv2
import os
import numpy as np

# Definir las rutas directamente en el script
encodings_path = r"C:\deep_learning\trabajo_final\output\encodings.pickle"
examples_path = r"C:\deep_learning\trabajo_final\examples"
detection_method = "hog"  # cnn, cambia a "hog" si prefieres mayor velocidad

# Cargar las codificaciones conocidas
print("[INFO] Cargando codificaciones...")
with open(encodings_path, "rb") as f:
    data = pickle.load(f)

# Obtener todas las imágenes de la carpeta examples
image_files = [f for f in os.listdir(examples_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# Función para cargar y redimensionar una imagen
def load_and_resize(image_path, height=100):
    image = cv2.imread(image_path)
    h, w = image.shape[:2]
    aspect_ratio = w / h
    new_width = int(height * aspect_ratio)
    return cv2.resize(image, (new_width, height))

# Crear un mosaico de imágenes
print("[INFO] Creando tira de imágenes...")
# Redimensionar todas las imágenes a la misma altura
thumbnails = [load_and_resize(os.path.join(examples_path, img)) for img in image_files]
# Hacer que todas tengan la misma altura y apilarlas horizontalmente
mosaic = np.hstack(thumbnails)

# Mostrar el mosaico y permitir selección con clic
def click_and_recognize(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # Calcular qué imagen fue clicada
        img_widths = [thumb.shape[1] for thumb in thumbnails]
        cumulative_width = 0
        for idx, width in enumerate(img_widths):
            if cumulative_width <= x < cumulative_width + width:
                selected_image = os.path.join(examples_path, image_files[idx])
                print(f"[INFO] Reconociendo rostros en '{image_files[idx]}'...")
                recognize_faces(selected_image)
                break
            cumulative_width += width

def recognize_faces(image_path):
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
    cv2.imshow("Resultado", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Mostrar la tira de imágenes
cv2.namedWindow("Tira de imágenes")
cv2.setMouseCallback("Tira de imágenes", click_and_recognize)

while True:
    # Mostrar el mosaico en la ventana
    cv2.imshow("Tira de imágenes", mosaic)
    # Verificar si la ventana ha sido cerrada manualmente
    if cv2.getWindowProperty("Tira de imágenes", cv2.WND_PROP_VISIBLE) < 1:
        print("[INFO] Ventana cerrada. Finalizando el programa...")
        break
    # Salir si se presiona la tecla Esc
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # Presionar 'Esc' para salir
        print("[INFO] Finalizando el programa por tecla Esc...")
        break

cv2.destroyAllWindows()
