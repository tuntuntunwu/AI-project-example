import os
import cv2
import numpy as np
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter


def translation(image, right=-50, down=20):
    shape = image.shape
    shape_size = shape[:2]
    # translation matrix [[1,0,-100], [0,1,-12]]
    M=np.array([[1,0,right], [0,1,down]], dtype=np.float32)
    
    out = cv2.warpAffine(image, M, shape_size[::-1], borderValue=255)
    # cv2.imshow("translation", out)
    return out

def rotation(image, angle=15, scale=1):
    shape = image.shape
    w, h = shape[:2]
    # rotation matrix
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, scale)

    out = cv2.warpAffine(image, M, (w, h), borderValue=255)
    # cv2.imshow("rotation", out)
    return out
   
def gaussian_noise(image, mean=0, var=0.001):
    image = np.array(image / 255, dtype=float)
    noise = np.random.normal(mean, var ** 0.5, image.shape)
    out = image + noise
    if out.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.
    out = np.clip(out, low_clip, 1.0)
    out = np.uint8(out*255)
    # cv2.imshow("gaussian_noise", out)
    return out

def affine_random(image, alpha=20):
    random_state = np.random.RandomState(None)
    shape = image.shape
    shape_size = shape[:2]

    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32([center_square + square_size, [center_square[0] + square_size, center_square[1] - square_size], center_square - square_size])
    pts2 = pts1 + random_state.uniform(-alpha, alpha, size=pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)

    out = cv2.warpAffine(image, M, shape_size[::-1], borderValue=255)
    # cv2.imshow("affine_random", out)
    return out

def elastic_random(image, alpha=10, sigma=0.8):
    random_state = np.random.RandomState(None)
    shape = image.shape
    shape_size = shape[:2]

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dz = np.zeros_like(dx)

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)), np.reshape(z, (-1, 1))

    out = map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)
    # cv2.imshow("elastic_random", out)
    return out

def elastic_affine_random(image, alpha_affine=20, alpha=10, sigma=0.8):
    random_state = np.random.RandomState(None)
    shape = image.shape
    shape_size = shape[:2]

    # affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32([center_square + square_size, [center_square[0]+square_size, center_square[1]-square_size], center_square - square_size])
    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, shape_size[::-1], borderValue=255)

    # elastic
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dz = np.zeros_like(dx)

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)), np.reshape(z, (-1, 1))

    out = map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)
    # cv2.imshow("elastic_affine_random", out)
    return out


if __name__ == '__main__':
    img = cv2.imread('./hehe/05_03131_051_A.png', 0)
    img1 = translation(img, right=-22, down=11)
    cv2.imwrite('t_2_05_03131_051_A.jpg', img1)

    img2 = affine_random(img, alpha=20)
    cv2.imwrite('a_5_05_03131_051_A.jpg', img2)


    # img = cv2.imread('./ori.jpg', 0)
    # img2 = translation(img, right=-50, down=20)
    # img3 = rotation(img, angle=15, scale=1)
    # img4 = gaussian_noise(img, mean=0, var=0.001)
    # img5 = affine_random(img, alpha=20)
    
    # img = cv2.imread('./ori.jpg', 1)
    # img6 = elastic_random(img, alpha=10, sigma=0.8)
    # img6 = cv2.cvtColor(img6, cv2.COLOR_BGR2GRAY)
    # img7 = elastic_affine_random(img, alpha_affine=20, alpha=10, sigma=0.8)
    # img7 = cv2.cvtColor(img7, cv2.COLOR_BGR2GRAY)

    # cv2.imwrite('img2.jpg', img2)
    # cv2.imwrite('img3.jpg', img3)
    # cv2.imwrite('img4.jpg', img4)
    # cv2.imwrite('img5.jpg', img5)
    # cv2.imwrite('img6.jpg', img6)
    # cv2.imwrite('img7.jpg', img7)
