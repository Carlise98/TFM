import os
import math
import numpy as np
import cv2
from PIL import Image
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

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

    path_images = "C:/Users/Carles/Documents/GitHub/LYT-Net/results/LOLv1"
    numero_archivos = len(os.listdir(path_images))

    # fig, ax = plt.subplots(layout='constrained')
    # width = 0.25  # the width of the bars
    # multiplier = 0

    for i in range(1,numero_archivos+1):

        
        # Rutas de las imágenes de entrada y salida
        LYT_path = "C:/Users/Carles/Documents/GitHub/LYT-Net/test/" + str(i) + ".png"  # Ruta de la primera imagen
        GSAD_path = "C:/Users/Carles/Documents/GitHub/GSAD/test/" + str(i) + "_normal.png"  # Ruta de la segunda imagen
        mean_path = "C:/Users/Carles/Documents/GitHub/TFM/images/LOLv1/mean/imagen_resultante_" + str(i) + ".png"  # Ruta de la imagen resultante
        target_path = "C:/Users/Carles/Documents/GitHub/TFM/images/LOLv1/target/" + str(i) + ".png"  # Ruta de la imagen resultante

        # Carga de imágenes
        LYT_image=cv2.imread(LYT_path, 1)
        GSAD_image=cv2.imread(GSAD_path, 1)
        mean_image=cv2.imread(mean_path, 1)
        target_image=cv2.imread(target_path, 1)

        # Cálculo de las métricas
        print('Imagen ' + str(i))
        print('SSIM-LYT')
        print(calculate_ssim(LYT_image, target_image))
        print('PSNR-LYT')
        print(calculate_psnr(LYT_image,target_image))
        
        print('SSIM-GSAD')
        print(calculate_ssim(GSAD_image, target_image))
        print('PSNR-GSAD')
        print(calculate_psnr(GSAD_image,target_image))

        print('SSIM-MEAN')
        print(calculate_ssim(mean_image, target_image))
        print('PSNR-MEAN')
        print(calculate_psnr(mean_image,target_image))
        print("-" * 20)

    #     offset = width * multiplier
    #     rects = ax.bar(45 + offset,
    #                     (calculate_ssim(LYT_image, target_image),
    #                     # calculate_psnr(LYT_image,target_image),
    #                     calculate_ssim(GSAD_image, target_image),
    #                     # calculate_psnr(GSAD_image,target_image),
    #                     calculate_ssim(mean_image, target_image))
    #                     # calculate_psnr(mean_image,target_image)]
    #                     , width, label='imagen' + str(i))
    #     ax.bar_label(rects, padding=3)
    #     multiplier += 1
    
    # ax.set_ylabel('Length (mm)')
    # ax.set_title('Penguin attributes by species')
    # # ax.set_xticks(15 + width, species)
    # ax.legend(loc='upper left', ncols=3)
    # ax.set_ylim(0, 1)
    # plt.show()