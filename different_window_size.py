import cv2
from main import Least_square_denoise

image = cv2.imread("./different_noise_level/noisy_image_0_01.png", 0)

LSD_1 = Least_square_denoise(5)
output_image_window_size_5 = LSD_1.LS_denoise(image)
cv2.imwrite("./different_window_size/output_image_window_size_5.png",
            output_image_window_size_5)

LSD_2 = Least_square_denoise(7)
output_image_window_size_7 = LSD_2.LS_denoise(image)
cv2.imwrite("./different_window_size/output_image_window_size_7.png",
            output_image_window_size_7)

LSD_3 = Least_square_denoise(9)
output_image_window_size_9 = LSD_3.LS_denoise(image)
cv2.imwrite("./different_window_size/output_image_window_size_9.png",
            output_image_window_size_9)