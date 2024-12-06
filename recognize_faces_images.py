# Importar los paquetes necesarios
import face_recognition
import pickle
import cv2
import os
import numpy as np

# Definir las rutas directamente en el script
encodings_path = r"C:\deep_learning\trabajo_final\output\encodingscnn.pickle"
examples_path = r"C:\deep_learning\trabajo_final\examples"
detection_method = "cnn"  # Usa "cnn" para mayor precisión si tienes GPU

# Cargar las codificaciones conocidas
print("[INFO] Cargando codificaciones...")
with open(encodings_path, "rb") as f:
    data = pickle.load(f)

# Obtener todas las imágenes de la carpeta examples
image_files = [f for f in os.listdir(examples_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# Función para cargar y redimensionar una imagen para la galeria
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

# Función para redimensionar una imagen al ancho y alto máximo permitido
def resize_to_max_dimensions(image, max_width=800, max_height=800):
    h, w = image.shape[:2]
    # Calcular las proporciones para ancho y alto
    width_ratio = max_width / w
    height_ratio = max_height / h
    # Escoger el menor ratio para mantener la proporción
    scaling_factor = min(width_ratio, height_ratio, 1.0) # No escalar si ya es menor
    new_width = int(w * scaling_factor)
    new_height = int(h * scaling_factor)
    return cv2.resize(image, (new_width, new_height))

def recognize_faces(image_path):
    # Cargar la imagen de entrada y convertirla de BGR a RGB
    image = cv2.imread(image_path)
    # Redimensionar la imagen al tamaño permitido  para evitar trabajar con imagenes muy grandes
    image = resize_to_max_dimensions(image, max_width=800, max_height=800)

    # Convertir a RGB
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detectar las ubicaciones de los rostros
    print("[INFO] Detectando rostros...")
    # number_of_times_to_upsample: Incrementar este parámetro permite detectar rostros más pequeños o lejanos. Usa 1 o 2 para mejorar la detección en imágenes de baja calidad.
    boxes = face_recognition.face_locations(rgb, model=detection_method, number_of_times_to_upsample=1)
    # num_jitters: ste parámetro aplica pequeñas perturbaciones a la imagen antes de calcular las codificaciones, generando representaciones más robustas.
    encodings = face_recognition.face_encodings(rgb, boxes, num_jitters=4)

    # Inicializar la lista de nombres
    names = []
    for encoding in encodings:
        #tolerance: Default 0.6 se configura en 0.5 mas extricto
        matches = face_recognition.compare_faces(data["encodings"], encoding, tolerance=0.5)
        name = "Desconocido"
        if True in matches:
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
        y = top - 15 if top - 15 > 15 else top + 15
        cv2.putText(image, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.50, (255, 255, 0), 2) 

        # Detectar los puntos de referencia faciales
        landmarks = face_recognition.face_landmarks(rgb, [(top, right, bottom, left)])
        
        # Dibujar los contornos faciales
        for landmark in landmarks:
            # Dibuja los contornos para cada grupo de puntos
            for feature, points in landmark.items():
                if feature in ["chin", "left_eyebrow", "right_eyebrow", "nose_bridge", "nose_tip", "top_lip", "bottom_lip"]:
                #if feature in ["chin"]:
                    # Conectar los puntos para formar el contorno
                    for i in range(1, len(points)):
                        cv2.line(image, points[i - 1], points[i], (192, 192, 192), 1) 
                    # Cierra el contorno para ojos y labios
                    #if feature in ["left_eye", "right_eye", "top_lip", "bottom_lip"]:
                        #cv2.line(image, points[-1], points[0], (192, 192, 192), 1)


    # Mostrar la imagen de salida
    cv2.imshow("Resultado", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Mostrar la tira de imágenes
cv2.namedWindow("Galeria")
cv2.setMouseCallback("Galeria", click_and_recognize)

while True:
    cv2.imshow("Galeria", mosaic)
    if cv2.getWindowProperty("Galeria", cv2.WND_PROP_VISIBLE) < 1:
        print("[INFO] Ventana cerrada. Finalizando el programa...")
        break
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # Esc para salir
        print("[INFO] Finalizando el programa por tecla Esc...")
        break

cv2.destroyAllWindows()
