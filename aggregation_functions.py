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

    return nueva_imagenos.chdir('images/LOLv2/Synthetic') 

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
    
    path_images = "C:/Users/Carles/Documents/GitHub/TFM/images/LOLv2/Synthetic/target"
    numero_archivos = len(os.listdir(path_images))
    # Directorio base
    base_path = os.getcwd()
    print(base_path)

    # Iterar sobre los archivos
    for i in range(1, numero_archivos+1):
        # Rutas de las imágenes de entrada
        image1_path = f"{base_path}/LYT/{i}.png"
        image2_path = f"{base_path}/GSAD/{i}.png"

        # Rutas de las imágenes de salida
        output_paths = [
            f"{base_path}/agregaciones/mean/imagen_mean_{i}.png",
            f"{base_path}/agregaciones/minimum/imagen_min_{i}.png",
            f"{base_path}/agregaciones/copula/imagen_copula_{i}.png",
            f"{base_path}/agregaciones/harmonic_mean/imagen_harmonic_{i}.png",
            f"{base_path}/agregaciones/geometric_mean/imagen_geometric_{i}.png",
            f"{base_path}/agregaciones/product/imagen_product_{i}.png"
        ]

        # Calcular las imágenes
        images = [
            average_images(image1_path, image2_path),
            min_images(image1_path, image2_path),
            min_sqrt_images(image1_path, image2_path),
            harmonic_mean_images(image1_path, image2_path),
            geometric_mean_images(image1_path, image2_path),
            algebraic_product_images(image1_path, image2_path)
        ]

        # Crear las carpetas de salida si no existen
        # Guardar las imágenes
        for output_path, image in zip(output_paths, images):
            output_folder = os.path.dirname(output_path)
            if not os.path.exists(output_folder):
                print('Creando...')
                os.makedirs(output_folder)
            cv2.imwrite(output_path, image)



