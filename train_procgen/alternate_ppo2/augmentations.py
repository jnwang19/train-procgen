import numpy as np
import tensorflow as tf
import PIL.Image as Image

def save_img(x, path):
    image_pil = Image.fromarray(x).convert('RGB')
    image_pil.save(path)

# Based on https://github.com/hongyi-zhang/mixup/blob/master/cifar/utils.py
def mixup_data(x, y, alpha=1.0):
    '''Compute the mixup data. Return mixed inputs and targets.'''
    
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    
    save_img(np.uint8(x[0]), "./before.png")
    index = np.random.shuffle(np.arange(x.shape[0]))

    mixed_x = lam * x + (1 - lam) * x[index,:]
    mixed_x = tf.squeeze(mixed_x).eval()

    y_a, y_b = y, y[index]
    mixed_y = lam * y_a + (1 - lam) * y_b
    save_img(np.uint8(mixed_x[0]), "./after.png")

    return mixed_x, mixed_y


