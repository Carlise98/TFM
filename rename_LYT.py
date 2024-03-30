import os

# Directorio donde se encuentran las imágenes
directorio_imagenes = r"C:\Users\Carles\Documents\GitHub\TFM\images\LOLv2\Synthetic\LYT/"

# Obtener la lista de nombres de archivo en el directorio
nombres_archivos = os.listdir(directorio_imagenes)

# Ordenar los nombres de archivo para asegurarse de que estén en el orden correcto
# nombres_archivos.sort()

# Contador para el nuevo nombre de archivo
contador = 1

# Iterar sobre los nombres de archivo y renombrarlos
for nombre_archivo in nombres_archivos:
    # Generar el nuevo nombre de archivo
    nuevo_nombre = '{:01d}.png'.format(contador)

    # Construir la ruta completa del archivo antiguo y nuevo
    antigua_ruta = os.path.join(directorio_imagenes, nombre_archivo)
    nueva_ruta = os.path.join(directorio_imagenes, nuevo_nombre)

    # Renombrar el archivo
    os.rename(antigua_ruta, nueva_ruta)

    # Incrementar el contador
    contador += 1

    # Salir del bucle después de renombrar 100 archivos
    if contador > 100:
        break
