# import the necessary packages
from imutils import paths
import face_recognition
import argparse
import pickle
import cv2
import os


# construct the argument parser and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--dataset", required=True,
# 	help="path to input directory of faces + images")
# ap.add_argument("-e", "--encodings", required=True,
# 	help="path to serialized db of facial encodings")
# ap.add_argument("-d", "--detection-method", type=str, default="cnn",
# 	help="face detection model to use: either `hog` or `cnn`")
# args = vars(ap.parse_args())

# Rutas al dataset y al directorio de salida
dataset_path = r"C:\deep_learning\trabajo_final\dataset"
output_path = r"C:\deep_learning\trabajo_final\output\encodingscnn.pickle"
detection_method = "cnn"  # cnn, pero puedes cambiar a "hog" si prefieres

# Función para redimensionar una imagen al ancho y alto máximo permitido
def resize_to_max_dimensions(image, max_width=600, max_height=600):
    h, w = image.shape[:2]
    # Calcular las proporciones para ancho y alto
    width_ratio = max_width / w
    height_ratio = max_height / h
    # Escoger el menor ratio para mantener la proporción
    scaling_factor = min(width_ratio, height_ratio, 1.0)  # No escalar si ya es menor
    new_width = int(w * scaling_factor)
    new_height = int(h * scaling_factor)
    return cv2.resize(image, (new_width, new_height))

# grab the paths to the input images in our dataset
print("[INFO] quantifying faces...")
# imagePaths = list(paths.list_images(args["dataset"]))
imagePaths = list(paths.list_images(dataset_path))
# initialize the list of known encodings and known names
knownEncodings = []
knownNames = []

# loop over the image paths
for (i, imagePath) in enumerate(imagePaths):
	# extract the person name from the image path
	print("[INFO] processing image {}/{}".format(i + 1,
		len(imagePaths)))
	name = imagePath.split(os.path.sep)[-2]
 
	# load the input image and convert it from BGR (OpenCV ordering)
	# to dlib ordering (RGB)
	image = cv2.imread(imagePath)
	# Redimensionar la imagen al ancho máximo permitido
	image = resize_to_max_dimensions(image, max_width=600, max_height=600)
    
	rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
 
	# detect the (x, y)-coordinates of the bounding boxes
	# corresponding to each face in the input image
	# boxes = face_recognition.face_locations(rgb, model=args["detection_method"])
	# number_of_times_to_upsample = 0: Más rápido, pero puede no detectar rostros pequeños. 1 o 2: Más lento, pero mejora la detección en imágenes donde los rostros son pequeños.
	boxes = face_recognition.face_locations(rgb, model=detection_method, number_of_times_to_upsample=1)
 	
 
	# compute the facial embedding for the face
	# num_jitters: 1 (por defecto): Más rápido, pero menos robusto a la variación. 2 o 3: Más lento, pero genera codificaciones más precisas al aplicar múltiples "perturbaciones".
	# encodings = face_recognition.face_encodings(rgb, boxes, num_jitters=1)
	encodings = face_recognition.face_encodings(rgb, boxes, num_jitters=10)
 
	# loop over the encodings
	for encoding in encodings:
		# add each encoding + name to our set of known names and
		# encodings
		knownEncodings.append(encoding)
		knownNames.append(name)
  
  # Guardar las codificaciones faciales y nombres en un archivo
print("[INFO] Serializando codificaciones...")
data = {"encodings": knownEncodings, "names": knownNames}
with open(output_path, "wb") as f:
    f.write(pickle.dumps(data))