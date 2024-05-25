import os
import math
import numpy as np
import cv2
from PIL import Image
import pandas as pd
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import torch
from torchmetrics.image import UniversalImageQualityIndex


def calculate_uiqi(img1, img2):
    """
    Calculate the Universal Image Quality Index (UIQI) between two images.
    
    Parameters:
    img1 (numpy.ndarray): The first image (reference image).
    img2 (numpy.ndarray): The second image (image to be compared).
    
    Returns:
    float: The UIQI value.
    """
    if img1.shape != img2.shape:
        raise ValueError("Input images must have the same dimensions.")
    
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    
    mean_x = np.mean(img1)
    mean_y = np.mean(img2)
    
    var_x = np.var(img1)
    var_y = np.var(img2)
    
    cov_xy = np.mean((img1 - mean_x) * (img2 - mean_y))
    
    numerator = 4 * mean_x * mean_y * cov_xy
    denominator = (var_x + var_y) * (mean_x**2 + mean_y**2)
    
    if denominator == 0:
        return 1 if numerator == 0 else 0
    
    return numerator / denominator

def calculate_uiqi_color(img1, img2):
    """
    Calculate the Universal Image Quality Index (UIQI) for color images by averaging
    the UIQI values of each channel (R, G, B).
    
    Parameters:
    img1 (numpy.ndarray): The first color image (reference image).
    img2 (numpy.ndarray): The second color image (image to be compared).
    
    Returns:
    float: The average UIQI value across all channels.
    """
    if img1.shape != img2.shape:
        raise ValueError("Input images must have the same dimensions and number of channels.")
    
    if len(img1.shape) == 2:  # Grayscale images
        return calculate_uiqi(img1, img2)
    elif len(img1.shape) == 3:  # Color images
        uiqi_r = calculate_uiqi(img1[:,:,0], img2[:,:,0])
        uiqi_g = calculate_uiqi(img1[:,:,1], img2[:,:,1])
        uiqi_b = calculate_uiqi(img1[:,:,2], img2[:,:,2])
        return (uiqi_r + uiqi_g + uiqi_b) / 3
    else:
        raise ValueError("Unsupported image format.")
    

def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)  
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')


if __name__ == "__main__":

    path_images = "C:/Users/Carles/Documents/GitHub/TFM/images/LOLv2/Synthetic/target"
    numero_archivos = len(os.listdir(path_images))

    ssim_lyt = []
    psnr_lyt = []
    uiqi_lyt = []

    ssim_gsad = []
    psnr_gsad = []
    uiqi_gsad = []

    ssim_copula = []
    psnr_copula = []
    uiqi_copula = []

    ssim_geometric = []
    psnr_geometric = []
    uiqi_geometric = []

    ssim_harmonic = []
    psnr_harmonic = []
    uiqi_harmonic = []

    ssim_mean = []
    psnr_mean = []
    uiqi_mean = []

    ssim_min = []
    psnr_min = []
    uiqi_min = []

    ssim_product = []
    psnr_product = []
    uiqi_product = []


    target_list = []
    geometric_list = []
    gsad_list = []
    lyt_list = []




    for i in range(1,numero_archivos+1):

        
        # Rutas de las imágenes de entrada y salida
        LYT_path = "C:/Users/Carles/Documents/GitHub/TFM/images/LOLv2/Synthetic/LYT/" + str(i) + ".png"  # Ruta de la primera imagen
        GSAD_path = "C:/Users/Carles/Documents/GitHub/TFM/images/LOLv2/Synthetic/GSAD/" + str(i) + ".png"  # Ruta de la segunda imagen

        copula_path = "C:/Users/Carles/Documents/GitHub/TFM/images/LOLv2/Synthetic/agregaciones/copula/imagen_copula_" + str(i) + ".png"  # Ruta de la imagen resultante
        geometric_path = "C:/Users/Carles/Documents/GitHub/TFM/images/LOLv2/Synthetic/agregaciones/geometric_mean/imagen_geometric_" + str(i) + ".png"  # Ruta de la imagen resultante
        harmonic_path = "C:/Users/Carles/Documents/GitHub/TFM/images/LOLv2/Synthetic/agregaciones/harmonic_mean/imagen_harmonic_" + str(i) + ".png"  # Ruta de la imagen resultante
        mean_path = "C:/Users/Carles/Documents/GitHub/TFM/images/LOLv2/Synthetic/agregaciones/mean/imagen_mean_" + str(i) + ".png"  # Ruta de la imagen resultante
        min_path = "C:/Users/Carles/Documents/GitHub/TFM/images/LOLv2/Synthetic/agregaciones/minimum/imagen_min_" + str(i) + ".png"  # Ruta de la imagen resultante
        product_path = "C:/Users/Carles/Documents/GitHub/TFM/images/LOLv2/Synthetic/agregaciones/product/imagen_product_" + str(i) + ".png"  # Ruta de la imagen resultante

        target_path = "C:/Users/Carles/Documents/GitHub/TFM/images/LOLv2/Synthetic/target/" + str(i) + ".png"  # Ruta de la imagen resultante

        # Carga de imágenes
        LYT_image=cv2.imread(LYT_path)
        GSAD_image=cv2.imread(GSAD_path, 1)

        copula_image=cv2.imread(copula_path, 1)
        geometric_image=cv2.imread(geometric_path, 1)
        harmonic_image=cv2.imread(harmonic_path, 1)
        mean_image=cv2.imread(mean_path, 1)
        min_image=cv2.imread(min_path, 1)
        product_image=cv2.imread(product_path, 1)

        target_image=cv2.imread(target_path)



        gsad_list.append(GSAD_image)
        lyt_list.append(LYT_image)
        geometric_list.append(geometric_image)
        target_list.append(target_image)
        


        # Cálculo de las métricas
        # print('Imagen ' + str(i))
        # print('SSIM-LYT')
        # print(calculate_ssim(LYT_image, target_image))
        # print('PSNR-LYT')
        # print(calculate_psnr(LYT_image,target_image))
        
        # print('SSIM-GSAD')
        # print(calculate_ssim(GSAD_image, target_image))
        # print('PSNR-GSAD')
        # print(calculate_psnr(GSAD_image,target_image))

        # print('SSIM-MEAN')
        # print(calculate_ssim(mean_image, target_image))
        # print('PSNR-MEAN')
        # print(calculate_psnr(mean_image,target_image))
        # print("-" * 20)
# --------------------------------------------------------------------------------------
        ssim_lyt.append(calculate_ssim(LYT_image, target_image))
        psnr_lyt.append(calculate_psnr(LYT_image, target_image))
        uiqi_lyt.append(calculate_uiqi_color(LYT_image, target_image))

        ssim_gsad.append(calculate_ssim(GSAD_image, target_image))
        psnr_gsad.append(calculate_psnr(GSAD_image, target_image))
        uiqi_gsad.append(calculate_uiqi_color(GSAD_image, target_image))

        ssim_copula.append(calculate_ssim(copula_image, target_image))
        psnr_copula.append(calculate_psnr(copula_image, target_image))
        uiqi_copula.append(calculate_uiqi_color(copula_image, target_image))

        ssim_geometric.append(calculate_ssim(geometric_image, target_image))
        psnr_geometric.append(calculate_psnr(geometric_image, target_image))
        uiqi_geometric.append(calculate_uiqi_color(geometric_image, target_image))

        ssim_harmonic.append(calculate_ssim(harmonic_image, target_image))
        psnr_harmonic.append(calculate_psnr(harmonic_image, target_image))
        uiqi_harmonic.append(calculate_uiqi_color(harmonic_image, target_image))

        ssim_mean.append(calculate_ssim(mean_image, target_image))
        psnr_mean.append(calculate_psnr(mean_image, target_image))
        uiqi_mean.append(calculate_uiqi_color(mean_image, target_image))

        ssim_min.append(calculate_ssim(min_image, target_image))
        psnr_min.append(calculate_psnr(min_image, target_image))
        uiqi_min.append(calculate_uiqi_color(min_image, target_image))

        ssim_product.append(calculate_ssim(product_image, target_image))
        psnr_product.append(calculate_psnr(product_image, target_image))
        uiqi_product.append(calculate_uiqi_color(product_image, target_image))
    
    
    # lyt_list = pd.DataFrame(data=lyt_list) 
    # target_list = pd.DataFrame(data=target_list) 

    # print(calculate_uiqi(lyt_list,target_list))
    
    # Graficar los resultados
    # plt.figure(figsize=(10, 6))

    # plt.subplot(2, 1, 1)
    # plt.plot(range(1, numero_archivos+1), ssim_lyt, label='SSIM LYT')
    # plt.plot(range(1, numero_archivos+1), ssim_gsad, label='SSIM GSAD')
    # plt.plot(range(1, numero_archivos+1), ssim_mean, label='SSIM MEAN')
    # plt.xlabel('Imagen')
    # plt.ylabel('Valor')
    # plt.title('SSIM')
    # plt.legend()

    # plt.subplot(2, 1, 2)
    # plt.plot(range(1, numero_archivos+1), psnr_lyt, label='PSNR LYT')
    # plt.plot(range(1, numero_archivos+1), psnr_gsad, label='PSNR GSAD')
    # plt.plot(range(1, numero_archivos+1), psnr_mean, label='PSNR MEAN')
    # plt.xlabel('Imagen')
    # plt.ylabel('Valor')
    # plt.title('PSNR')
    # plt.legend()

    # plt.tight_layout()
    # plt.show()
    # Calcular la media de los valores de cada métrica
    # media_ssim_lyt = np.mean(ssim_lyt)
    # media_psnr_lyt = np.mean(psnr_lyt)
    media_uiqi_lyt = np.mean(uiqi_lyt)

    # media_ssim_gsad = np.mean(ssim_gsad)
    # media_psnr_gsad = np.mean(psnr_gsad)
    media_uiqi_gsad = np.mean(uiqi_gsad)

    # media_ssim_copula = np.mean(ssim_copula)
    # media_psnr_copula = np.mean(psnr_copula)
    media_uiqi_copula = np.mean(uiqi_geometric)

    # media_ssim_geometric = np.mean(ssim_geometric)
    # media_psnr_geometric = np.mean(psnr_geometric)
    media_uiqi_geometric = np.mean(uiqi_geometric)

    # media_ssim_harmonic = np.mean(ssim_harmonic)
    # media_psnr_harmonic = np.mean(psnr_harmonic)
    media_uiqi_harmonic = np.mean(uiqi_harmonic)

    # media_ssim_mean = np.mean(ssim_mean)
    # media_psnr_mean = np.mean(psnr_mean)
    media_uiqi_mean = np.mean(uiqi_mean)

    # media_ssim_min = np.mean(ssim_min)
    # media_psnr_min = np.mean(psnr_min)
    media_uiqi_min = np.mean(uiqi_min)

    # # media_ssim_product = np.mean(ssim_product)
    # media_psnr_product = np.mean(psnr_product)
    media_uiqi_product = np.mean(uiqi_product)


    # Imprimir la media de los valores de cada métrica
    # print('Media SSIM LYT:', media_ssim_lyt)
    # print('Media PSNR LYT:', media_psnr_lyt)
    # print('Media SSIM GSAD:', media_ssim_gsad)
    # print('Media PSNR GSAD:', media_psnr_gsad)

    # print('Media SSIM COPULA:', media_ssim_copula)
    # print('Media PSNR COPULA:', media_psnr_copula)
    # print('Media SSIM GEOMETRIC:', media_ssim_geometric)
    # print('Media PSNR GEOMETRIC:', media_psnr_geometric)
    # print('Media SSIM HARMONIC:', media_ssim_harmonic)
    # print('Media PSNR HARMONIC:', media_psnr_harmonic)
    # print('Media SSIM MEAN:', media_ssim_mean)
    # print('Media PSNR MEAN:', media_psnr_mean)
    # print('Media SSIM MIN:', media_ssim_min)
    # print('Media PSNR MIN:', media_psnr_min)
    # print('Media SSIM PRODUCT:', media_ssim_product)
    # print('Media PSNR PRODUCT:', media_psnr_product)


    # Graficar las medias de las métricas
    metricas_ssim = ['SSIM LYT', 'SSIM GSAD','SSIM COPULA', 'SSIM GEOMETRIC', 'SSIM HARMONIC', 'SSIM MEAN', 'SSIM MIN', 'SSIM PRODUCT' ]
    metricas_psnr = ['PSNR LYT', 'PSNR GSAD','PSNR COPULA', 'PSNR GEOMETRIC', 'PSNR HARMONIC', 'PSNR MEAN', 'PSNR MIN', 'PSNR PRODUCT']
    metricas_uiqi = ['UIQI LYT', 'UIQI GSAD','UIQI COPULA', 'UIQI GEOMETRIC', 'UIQI HARMONIC', 'UIQI MEAN', 'UIQI MIN', 'UIQI PRODUCT']

    # medias_ssim = [media_ssim_lyt, media_ssim_gsad, media_ssim_copula, media_ssim_geometric, media_ssim_harmonic, media_ssim_mean, media_ssim_min, media_ssim_product]
    # medias_psnr = [media_psnr_lyt, media_psnr_gsad, media_psnr_copula, media_psnr_geometric, media_psnr_harmonic, media_psnr_mean, media_psnr_min, media_psnr_product]
    medias_uiqi = [media_uiqi_lyt, media_uiqi_gsad, media_uiqi_copula, media_uiqi_geometric, media_uiqi_harmonic, media_uiqi_mean, media_uiqi_min, media_uiqi_product]
    
    # plt.bar(metricas, medias)
    # plt.xlabel('Métricas')
    # plt.ylabel('Media')
    # plt.title('Medias de Métricas')
    # plt.xticks(rotation=45, ha='right')
    # plt.tight_layout()
    # plt.show()


    # Graficar las medias de las métricas SSIM en una gráfica
    plt.figure(figsize=(10, 6))
    

    # plt.subplot(3, 1, 1)
    # bars = plt.bar(metricas_ssim, medias_ssim, color=['blue', 'green', 'purple'])
    # plt.xlabel('Métricas SSIM')
    # plt.ylabel('Media')
    # plt.title('Medias de Métricas SSIM')
    # plt.xticks(rotation=45, ha='right')
    # plt.grid(True)
    
    # for bar, media in zip(bars, medias_ssim):
    #     plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{media:.2f}', ha='center', va='bottom')

    # plt.subplot(3, 1, 2)
    # plt.bar(metricas_psnr, medias_psnr, color=['orange', 'red', 'brown'])
    # plt.xlabel('Métricas PSNR')
    # plt.ylabel('Media')
    # plt.title('Medias de Métricas PSNR')
    # plt.xticks(rotation=45, ha='right')
    # plt.grid(True)

    # for bar, media in zip(bars, medias_psnr):
    #     plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{media:.2f}', ha='center', va='bottom')

    plt.subplot(1, 1, 1)
    bars = plt.bar(metricas_uiqi, medias_uiqi, color=['orange', 'red', 'brown'])
    plt.xlabel('Métricas UIQI')
    plt.ylabel('Media')
    plt.title('Medias de Métricas UIQI')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True)

    for bar, media in zip(bars, medias_uiqi):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{media:.2f}', ha='center', va='bottom')
   
    plt.tight_layout()
    plt.show()