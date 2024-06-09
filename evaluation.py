import cv2
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

# truth_image = cv2.imread('./sw_try/input.png', 0)
# noise_image = cv2.imread('./sw_try/noise.png', 0)
# denoise_image = cv2.imread('./sw_try/after_denoise.png', 0)
#
# psnr_1 = compare_psnr(truth_image, noise_image)
# psnr_2 = compare_psnr(truth_image, denoise_image)
#
# print(f"加入椒盐噪声后的峰值信噪比为：{psnr_1:.2f}")
# print(f"最小二乘降噪后的峰值信噪比为：{psnr_2:.2f}")

# truth_image = cv2.imread('./input.png', 0)
# noise_image_0_01 = cv2.imread('./different_noise_level/noisy_image_0_01.png', 0)
# noise_image_0_05 = cv2.imread('./different_noise_level/noisy_image_0_05.png', 0)
# denoise_image_0_01 = cv2.imread('./different_noise_level/output_image_0_01.png', 0)
# denoise_image_0_05 = cv2.imread('./different_noise_level/output_image_0_05.png', 0)
#
# psnr_1 = compare_psnr(truth_image, noise_image_0_01)
# psnr_2 = compare_psnr(truth_image, noise_image_0_05)
# psnr_3 = compare_psnr(truth_image, denoise_image_0_01)
# psnr_4 = compare_psnr(truth_image, denoise_image_0_05)

# print(f"{psnr_1:.3f}")
# print(f"{psnr_2:.3f}")
# print(f"{psnr_3:.3f}")
# print(f"{psnr_4:.3f}")

# truth_image = cv2.imread('./input.png', 0)
#
# denoise_image_5 = cv2.imread('./different_window_size/output_image_window_size_5.png', 0)
# denoise_image_7 = cv2.imread('./different_window_size/output_image_window_size_7.png', 0)
# denoise_image_9 = cv2.imread('./different_window_size/output_image_window_size_9.png', 0)
#
# psnr_1 = compare_psnr(truth_image, denoise_image_5)
# psnr_2 = compare_psnr(truth_image, denoise_image_7)
# psnr_3 = compare_psnr(truth_image, denoise_image_9)
#
# print(f"{psnr_1:.3f}")
# print(f"{psnr_2:.3f}")
# print(f"{psnr_3:.3f}")

truth_image = cv2.imread('./input.png', 0)

denoise_image_1 = cv2.imread('./SW_LSD/output_image.png', 0)
denoise_image_2 = cv2.imread('./SW_LSD/output_image_side_window.png', 0)

psnr_1 = compare_psnr(truth_image, denoise_image_1)
psnr_2 = compare_psnr(truth_image, denoise_image_2)

print(f"{psnr_1:.3f}")
print(f"{psnr_2:.3f}")