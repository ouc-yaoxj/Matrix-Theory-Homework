import cv2
import numpy as np
from main import add_salt_and_pepper_noise, Least_square_denoise

image = cv2.imread('/mnt/4T/yxj/Image_denoise_by_least_square/1.jpg', 0)
cv2.imwrite("./different_noise_level/input.png", image)

# 加入比例为0.01的椒盐噪声
noisy_image_001 = add_salt_and_pepper_noise(image, 0.01, 0.01)
cv2.imwrite("./different_noise_level/noisy_image_0_01.png", noisy_image_001)

# 加入比例为0.05的椒盐噪声
noisy_image_005 = add_salt_and_pepper_noise(image, 0.05, 0.05)
cv2.imwrite("./different_noise_level/noisy_image_0_05.png", noisy_image_005)

LSD = Least_square_denoise(7)
output_image_001 = LSD.LS_denoise(noisy_image_001)
cv2.imwrite("./different_noise_level/output_image_0_01.png", output_image_001)

output_image_005 = LSD.LS_denoise(noisy_image_005)
cv2.imwrite("./different_noise_level/output_image_0_05.png", output_image_005)
