import cv2
import numpy as np
import os

def load_images(image1_path, image2_path):
    img1 = cv2.imread(image1_path)
    img2 = cv2.imread(image2_path)

     # Verificar que las dimensiones de las imágenes sean iguales
    if img1.shape != img2.shape:
        raise ValueError("Las imágenes tienen dimensiones diferentes")
    
    return img1,img2

def scalar(img):
    img_norm = img.astype(np.float32) / 255.0
    return img_norm


def min_images(image1_path, image2_path):
    
    img1,img2 = load_images(image1_path,image2_path)

    # Convertir las imágenes a valores de píxeles flotantes y normalizar entre 0 y 1
    img1_norm = scalar(img1)
    img2_norm = scalar(img2)

    # Calcular el píxel de menor valor
    min_imagen = np.minimum(img1_norm, img2_norm)

    # Escalar nuevamente a valores entre 0 y 255
    min_imagen = (min_imagen * 255).astype(np.uint8)

    return min_imagen
    
def average_images(image1_path,image2_path):
    
    img1,img2 = load_images(image1_path,image2_path)

    # Convertir las imágenes a valores de píxeles flotantes y normalizar entre 0 y 1
    img1_norm = scalar(img1)
    img2_norm = scalar(img2)

    # Calcular la media de los valores de píxeles
    mean_imagen = (img1_norm + img2_norm) / 2.0

    # Escalar nuevamente a valores entre 0 y 255
    mean_imagen = (mean_imagen * 255).astype(np.uint8)

    return mean_imagen

def min_sqrt_images(image1_path, image2_path):

    img1,img2 = load_images(image1_path,image2_path)

    # Convertir las imágenes a valores de píxeles flotantes y normalizar entre 0 y 1
    img1_norm = scalar(img1)
    img2_norm = scalar(img2)

    # Calcular el mínimo entre la raíz cuadrada de uno multiplicado por el otro y viceversa
    nueva_imagen = np.minimum(np.sqrt(img1_norm) * img2_norm, np.sqrt(img2_norm) * img1_norm)

    # Escalar nuevamente a valores entre 0 y 255
    nueva_imagen = (nueva_imagen * 255).astype(np.uint8)

    return nueva_imagen

def harmonic_mean_images(image1_path, image2_path):

    img1,img2 = load_images(image1_path,image2_path)

    # Convertir las imágenes a valores de píxeles flotantes y normalizar entre 0 y 1
    img1_norm = scalar(img1)
    img2_norm = scalar(img2)

    # Calcular el píxel resultante según la regla especificada
    nueva_imagen = np.where((img1_norm == 0) | (img2_norm == 0), 0, 2 / ((1 / img1_norm) + (1 / img2_norm)))

    # Escalar nuevamente a valores entre 0 y 255
    nueva_imagen = (nueva_imagen * 255).astype(np.uint8)

    return nueva_imagen

'''
    Media geometrica
'''

def geometric_mean_images(image1_path, image2_path):

    img1,img2 = load_images(image1_path,image2_path)

    # Convertir las imágenes a valores de píxeles flotantes y normalizar entre 0 y 1
    img1_norm = scalar(img1)
    img2_norm = scalar(img2)

    # Calcular el píxel resultante según la regla especificada
    nueva_imagen = np.sqrt(np.multiply(img1_norm, img2_norm))

    # Escalar nuevamente a valores entre 0 y 255
    nueva_imagen = (nueva_imagen * 255).astype(np.uint8)

    return nueva_imagen

def algebraic_product_images(image1_path,image2_path):

    img1,img2 = load_images(image1_path,image2_path)

    # Convertir las imágenes a valores de píxeles flotantes y normalizar entre 0 y 1
    img1_norm = scalar(img1)
    img2_norm = scalar(img2)

    # Calcular el píxel resultante según la regla especificada
    nueva_imagen = np.multiply(img1_norm, img2_norm)

    # Escalar nuevamente a valores entre 0 y 255
    nueva_imagen = (nueva_imagen * 255).astype(np.uint8)

    return nueva_imagen

    
   
if __name__ == "__main__":
    os.chdir('../') 
    print(os.getcwd())
    path_images = "C:/Users/Carles/Documents/GitHub/LYT-Net/results/LOLv1"
    numero_archivos = len(os.listdir(path_images))

    for i in range(1,numero_archivos+1):
        # Rutas de las imágenes de entrada y salida
        image1_path = "C:/Users/Carles/Documents/GitHub/LYT-Net/test/" + str(i) + ".png"  # Ruta de la primera imagen
        image2_path = "C:/Users/Carles/Documents/GitHub/GSAD/test/" + str(i) + ".png"  # Ruta de la segunda imagen
        average_path = "C:/Users/Carles/Documents/GitHub/TFM/images/mean/imagen_mean_" + str(i) + ".png"  # Ruta de la imagen resultante
        min_path = "C:/Users/Carles/Documents/GitHub/TFM/images/minimum/imagen_min_" + str(i) + ".png"  # Ruta de la imagen resultante
        copula_path = "C:/Users/Carles/Documents/GitHub/TFM/images/copula/imagen_copula_" + str(i) + ".png"  # Ruta de la imagen resultante
        harmonic_path = "C:/Users/Carles/Documents/GitHub/TFM/images/harmonic_mean/imagen_harmonic_" + str(i) + ".png"  # Ruta de la imagen resultante
        geometric_path = "C:/Users/Carles/Documents/GitHub/TFM/images/geometric_mean/imagen_geometric_" + str(i) + ".png"  # Ruta de la imagen resultante
        product_path = "C:/Users/Carles/Documents/GitHub/TFM/images/product/imagen_product_" + str(i) + ".png"  # Ruta de la imagen resultante

        # Llamar a la función para crear la imagen promedio
        mean_image = average_images(image1_path, image2_path)
        min_image = min_images(image1_path, image2_path)
        copula_image = min_sqrt_images(image1_path, image2_path)
        harmonic_image = harmonic_mean_images(image1_path, image2_path)
        geometric_image = geometric_mean_images(image1_path, image2_path)
        product_image = algebraic_product_images(image1_path, image2_path)

        cv2.imwrite(average_path, mean_image)  # Guardar la nueva imagen
        cv2.imwrite(min_path, min_image)       # Guardar la nueva imagen
        cv2.imwrite(copula_path, copula_image)  # Guardar la nueva imagen
        cv2.imwrite(harmonic_path, harmonic_image)  # Guardar la nueva imagen
        cv2.imwrite(geometric_path, geometric_image)  # Guardar la nueva imagen
        cv2.imwrite(product_path, product_image)  # Guardar la nueva imagen




