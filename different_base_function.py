import cv2
from main import Least_square_denoise

image = cv2.imread("./different_noise_level/noisy_image_0_01.png", 0)

LSD_1 = Least_square_denoise(7)
output_image_highest_degree_1 = LSD_1.LS_denoise(image)
cv2.imwrite("./different_base_function/output_image_highest_degree_1.png",
            output_image_highest_degree_1)

LSD_2 = Least_square_denoise(7, num_base_function=6)
output_image_highest_degree_2 = LSD_2.LS_denoise(image)
cv2.imwrite("./different_base_function/output_image_highest_degree_2.png",
            output_image_highest_degree_2)