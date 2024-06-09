import cv2
import numpy as np
import matplotlib.pyplot as plt

def add_gaussian_noise(image, mean, var):
    image = image / 255
    noise = np.random.normal(mean, var ** 0.5, image.shape)
    output_image = image + noise
    out_image = np.clip(output_image, 0.0, 1.0) * 255
    out_image = out_image.astype(np.uint8)
    return out_image

def add_salt_and_pepper_noise(image, salt_prob, pepper_prob):
    """
    向图像添加椒盐噪声
    :param image: 输入的图像
    :param salt_prob: 盐噪声概率
    :param pepper_prob: 椒噪声概率
    :return: 添加噪声后的图像
    """
    output = np.copy(image)
    # 添加盐噪声
    num_salt = np.ceil(salt_prob * image.size)
    coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in image.shape]
    output[coords[0], coords[1]] = 255

    # 添加椒噪声
    num_pepper = np.ceil(pepper_prob * image.size)
    coords = [np.random.randint(0, i - 1, int(num_pepper))
              for i in image.shape]
    output[coords[0], coords[1]] = 0
    return output

class Least_square_denoise:
    def __init__(self, d, num_base_function=3):
        self.d = d
        self.num_base_function = num_base_function

    def get_input_coordinate_array(self, h_start, h_end, w_start, w_end):
        coordinate_array = np.zeros(((h_end - h_start + 1) * (w_end - w_start + 1), 2))
        n = 0
        for i in range(h_start, h_end + 1):
            for j in range(w_start, w_end + 1):
                coordinate_array[n, 0] = j
                coordinate_array[n, 1] = i
                n += 1

        return coordinate_array

    def get_base_vector_group(self, input_coordinate_array):
        A = np.zeros((len(input_coordinate_array), self.num_base_function))
        x = input_coordinate_array[:, 0]
        y = input_coordinate_array[:, 1]

        if self.num_base_function == 6:
            A[:, 0] = x ** 2
            A[:, 1] = y ** 2
            A[:, 2] = x * y
            A[:, 3] = x
            A[:, 4] = y
            A[:, 5] = 1
        elif self.num_base_function == 3:
            A[:, 0] = x
            A[:, 1] = y
            A[:, 2] = 1

        return A

    def small_area_LSF(self, small_area, A):
        lambda_ = small_area.ravel()
        fitting_func_coefficient = np.linalg.inv(A.T @ A) @ A.T @ lambda_
        central_value = fitting_func_coefficient[-1]

        return central_value

    def choose_side_window(self, side_window_mode, small_area):
        if side_window_mode == "L":
            sw_small_area = small_area[:, 0:self.d // 2 + 1]
            sw_coord = self.get_input_coordinate_array(h_start=0 - self.d // 2, h_end=self.d // 2,
                                                   w_start=0 - self.d // 2, w_end=0)
            sw_A = self.get_base_vector_group(sw_coord)
        elif side_window_mode == "R":
            sw_small_area = small_area[:, self.d // 2:]
            sw_coord = self.get_input_coordinate_array(h_start=0 - self.d // 2, h_end=self.d // 2,
                                                   w_start=0, w_end=self.d // 2)
            sw_A = self.get_base_vector_group(sw_coord)
        elif side_window_mode == "U":
            sw_small_area = small_area[0:self.d // 2 + 1, :]
            sw_coord = self.get_input_coordinate_array(h_start=0 - self.d // 2, h_end=0,
                                                   w_start=0 - self.d // 2, w_end=self.d // 2)
            sw_A = self.get_base_vector_group(sw_coord)
        elif side_window_mode == "D":
            sw_small_area = small_area[self.d // 2:, :]
            sw_coord = self.get_input_coordinate_array(h_start=0, h_end=self.d // 2,
                                                   w_start=0 - self.d // 2, w_end=self.d // 2)
            sw_A = self.get_base_vector_group(sw_coord)
        elif side_window_mode == "NW":
            sw_small_area = small_area[0:self.d // 2 + 1, 0:self.d // 2 + 1]
            sw_coord = self.get_input_coordinate_array(h_start=0 - self.d // 2, h_end=0,
                                                   w_start=0 - self.d // 2, w_end=0)
            sw_A = self.get_base_vector_group(sw_coord)
        elif side_window_mode == "NE":
            sw_small_area = small_area[0:self.d // 2 + 1, self.d // 2:]
            sw_coord = self.get_input_coordinate_array(h_start=0 - self.d // 2, h_end=0,
                                                   w_start=0, w_end=self.d // 2)
            sw_A = self.get_base_vector_group(sw_coord)
        elif side_window_mode == "SW":
            sw_small_area = small_area[self.d // 2:, 0:self.d // 2 + 1]
            sw_coord = self.get_input_coordinate_array(h_start=0, h_end=self.d // 2,
                                                   w_start=0 - self.d // 2, w_end=0)
            sw_A = self.get_base_vector_group(sw_coord)
        elif side_window_mode == "SE":
            sw_small_area = small_area[self.d // 2:, self.d // 2:]
            sw_coord = self.get_input_coordinate_array(h_start=0, h_end=self.d // 2,
                                                   w_start=0, w_end=self.d // 2)
            sw_A = self.get_base_vector_group(sw_coord)
        return sw_small_area, sw_A

    def LS_denoise(self, image):
        H, W = image.shape
        A = self.get_base_vector_group(
            self.get_input_coordinate_array(h_start=0 - self.d // 2, h_end=self.d // 2,
                                            w_start=0 - self.d // 2, w_end=self.d // 2)
        )
        output = image.copy()
        image_padding = np.pad(image, (self.d // 2, self.d // 2), "edge")
        for i in range(H):
            for j in range(W):
                small_area = image_padding[i:i+self.d, j:j+self.d]
                central_value = self.small_area_LSF(small_area=small_area, A=A)
                output[i, j] = central_value

        output = output.astype(np.uint8)
        return output

    def SW_LS_denoise(self, image):
        H, W = image.shape
        output = image.copy()
        image_padding = np.pad(image, (self.d // 2, self.d // 2), "edge")
        for i in range(H):
            for j in range(W):
                small_area = image_padding[i:i + self.d, j:j + self.d]
                dis = []
                dis_value_dict = {}
                for side_window_mode in ["L", "R", "U", "D", "NW", "NE", "SW", "SE"]:
                    sw_small_area, sw_A = self.choose_side_window(side_window_mode=side_window_mode,
                                                                  small_area=small_area)
                    tmp_central_value = self.small_area_LSF(small_area=sw_small_area, A=sw_A)
                    dis.append(abs(tmp_central_value - image[i, j]))
                    dis_value_dict[abs(tmp_central_value - image[i, j])] = tmp_central_value
                output[i, j] = dis_value_dict[min(dis)]

        output = output.astype(np.uint8)
        return output

if __name__ == "__main__":
    # image = cv2.imread('/mnt/4T/yxj/digital_image_processing/Assignment_1/image/RGB.jpg', 0)
    image = cv2.imread('/mnt/4T/yxj/Image_denoise_by_least_square/1.jpg', 0)
    cv2.imwrite("./sw_try/input.png", image)
    noisy_image = add_salt_and_pepper_noise(image, 0.01, 0.01)
    # noisy_image = add_gaussian_noise(image, 0, 0.01)
    cv2.imwrite("./sw_try/noise.png", noisy_image)

    LSD = Least_square_denoise(7)
    # output_image = LSD.LS_denoise(noisy_image)
    output_image = LSD.SW_LS_denoise(noisy_image)
    # print(output_image)
    cv2.imwrite("./sw_try/after_denoise.png", output_image)
    # cv2.waitKey(0)