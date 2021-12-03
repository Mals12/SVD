import numpy as np
from PIL import Image

def rebuild2(u,sigma,v,k):
    "k: primera dimensión k"
    uk = u[:,:k] # Primeras k columnas
    sigma_k=np.diag(sigma[:k])  # Los primeros k valores singulares
    vk = v[:k,:] # Primeras k líneas

    dot1 = np.dot(uk, sigma_k)
    return np.dot(dot1,vk)  # uk*Σk*vk

pic = Image.open('/Users/zhengwei/Pictures/zxt.jpg', 'r')
# Conversión de escala de grises
pic = pic.convert("L")
# Convertir a matriz
pic_arr = np.array(pic)
print(pic_arr.shape)
# SVD
u, sigma, v = np.linalg.svd(pic_arr[:, :])
print(sigma.shape)
# Una décima dimensión, las primeras 64, un total de 640 dimensiones
L = rebuild2(u, sigma, v,k=64)
Image.fromarray(L).show()
# Una quinta dimensión, las primeras 128, un total de 640 dimensiones
L = rebuild2(u, sigma, v,k=128)
Image.fromarray(L).show()
