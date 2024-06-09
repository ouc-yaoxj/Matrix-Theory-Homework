import cv2
from main import Least_square_denoise

image = cv2.imread("./different_noise_level/noisy_image_0_01.png", 0)
LSD = Least_square_denoise(7)
output_image = LSD.LS_denoise(image)
cv2.imwrite("./SW_LSD/output_image.png",
            output_image)
output_image_side_window = LSD.SW_LS_denoise(image)
cv2.imwrite("./SW_LSD/output_image_side_window.png",
            output_image_side_window)