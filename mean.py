from PIL import Image
import os

def transform_images(image1_path, image2_path, average_path,min_path):
    # Abrir las imágenes
    image1 = Image.open(image1_path)
    image2 = Image.open(image2_path)

    # Verificar si las dimensiones son iguales
    if image1.size != image2.size:
        raise ValueError("Las imágenes deben tener las mismas dimensiones")

    # Obtener las dimensiones de la imagen
    width, height = image1.size

    # Crear una nueva imagen para almacenar el resultado
    result_image = Image.new('RGB', (width, height))
    min_image = Image.new('RGB', (width, height))

    # Calcular la media de los píxeles
    for x in range(width):
        for y in range(height):
            # Obtener los píxeles de ambas imágenes
            pixel1 = image1.getpixel((x, y))
            pixel2 = image2.getpixel((x, y))

            # Calcular la media de cada canal RGB
            averaged_pixel = tuple((a + b) // 2 for a, b in zip(pixel1, pixel2))
            # Escoger el píxel de menor valor
            pixel_menor = tuple(min(c1, c2) for c1, c2 in zip(pixel1, pixel2))

            # Establecer el píxel en la nueva imagen
            result_image.putpixel((x, y), averaged_pixel)
            min_image.putpixel((x, y), pixel_menor)

    # Guardar la imagen resultante
    result_image.save(average_path)
    min_image.save(min_path)
    print("Imagen creada con éxito:", average_path)

if __name__ == "__main__":

    path_images = "C:/Users/Carles/Documents/GitHub/LYT-Net/results/LOLv1"
    numero_archivos = len(os.listdir(path_images))

    for i in range(1,numero_archivos+1):
        # Rutas de las imágenes de entrada y salida
        image1_path = "C:/Users/Carles/Documents/GitHub/LYT-Net/test/" + str(i) + ".png"  # Ruta de la primera imagen
        image2_path = "C:/Users/Carles/Documents/GitHub/GSAD/test/" + str(i) + "_normal.png"  # Ruta de la segunda imagen
        average_path = "C:/Users/Carles/Documents/GitHub/TFM/images/mean/imagen_mean_" + str(i) + ".png"  # Ruta de la imagen resultante
        min_path = "C:/Users/Carles/Documents/GitHub/TFM/images/minimum/imagen_min_" + str(i) + ".png"  # Ruta de la imagen resultante

        # Llamar a la función para crear la imagen promedio
        transform_images(image1_path, image2_path, average_path, min_path)




