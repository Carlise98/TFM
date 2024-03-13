from PIL import Image

def average_images(image1_path, image2_path, output_path):
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

    # Calcular la media de los píxeles
    for x in range(width):
        for y in range(height):
            # Obtener los píxeles de ambas imágenes
            pixel1 = image1.getpixel((x, y))
            pixel2 = image2.getpixel((x, y))

            # Calcular la media de cada canal RGB
            averaged_pixel = tuple((a + b) // 2 for a, b in zip(pixel1, pixel2))

            # Establecer el píxel en la nueva imagen
            result_image.putpixel((x, y), averaged_pixel)

    # Guardar la imagen resultante
    result_image.save(output_path)
    print("Imagen creada con éxito:", output_path)

if __name__ == "__main__":
    # Rutas de las imágenes de entrada y salida
    image1_path = "C:/Users/Carles/Documents/GitHub/LYT-Net/results/LOLv1/1.png"  # Ruta de la primera imagen
    image2_path = "C:/Users/Carles/Documents/GitHub/GSAD/experiments/lolv1_test_240226_194832/results/output/1_normal.png"  # Ruta de la segunda imagen
    output_path = "C:/Users/Carles/Desktop/imagenes/imagen_resultante.png"  # Ruta de la imagen resultante

    # Llamar a la función para crear la imagen promedio
    average_images(image1_path, image2_path, output_path)
