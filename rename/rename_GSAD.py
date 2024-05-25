import os

# Directorio donde se encuentran los archivos
base_path = os.getcwd()

# f"{base_path}/images\LOLv2\Synthetic\GSAD/"
directorio = f"{base_path}/images\LOLv2\Synthetic\GSAD/"

# Obtener la lista de nombres de archivo en el directorio
nombres_archivos = os.listdir(directorio)

# Iterar sobre los nombres de archivo y quitar '_normal' del nombre si está presente
for nombre_archivo in nombres_archivos:
    # Comprobar si '_normal' está en el nombre del archivo
    if '_normal' in nombre_archivo:
        # Generar el nuevo nombre de archivo sin '_normal'
        nuevo_nombre = nombre_archivo.replace('_normal', '')

        # Construir la ruta completa del archivo antiguo y nuevo
        antigua_ruta = os.path.join(directorio, nombre_archivo)
        nueva_ruta = os.path.join(directorio, nuevo_nombre)

        # Renombrar el archivo
        os.rename(antigua_ruta, nueva_ruta)
