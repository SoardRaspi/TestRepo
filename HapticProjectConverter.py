import numpy as np
from PIL import Image as im
import cv2 as cv

sizes = [256, 272, 288, 304, 320, 336, 352, 368, 384, 400, 416, 432, 448, 464, 480, 496, 512, 528, 544, 560, 576, 592, 608, 624, 640, 656, 672, 688, 704, 720, 736, 752, 768, 784, 800, 816, 832, 848, 864, 880, 896, 912, 928, 944, 960, 976, 992, 1008, 1024]

# prev = [0, 0]

def find_integral_coordinates(x1, y1, distance, slope):
    # Handle the special case where the slope is vertical or horizontal
    if slope == float('inf'):  # Vertical slope
        x2 = x1
        y2 = y1 + distance if distance > 0 else y1 - distance
    elif slope == 0:  # Horizontal slope
        x2 = x1 + distance if distance > 0 else x1 - distance
        y2 = y1
    else:
        # Calculate the squared terms in the Pythagorean theorem
        a_squared = distance ** 2 / (1 + slope ** 2)
        b_squared = distance ** 2 - a_squared

        # Calculate the rise and run
        rise = int(a_squared ** 0.5)
        run = int(b_squared ** 0.5)

        # Adjust the rise and run based on the sign of slope
        rise = rise if slope > 0 else -rise
        run = run if slope > 0 else -run

        # Calculate the new coordinates
        x2 = x1 + run
        y2 = y1 + rise

    # print([x2, y2])
    return x2, y2

def start_sequence(sequence_m, original_size, original_beta, original_point):
    print("sequence:", sequence_m)
    # index = int(input("Enter the required index: "))
    # sequence = sequence[index]

    if True:
        size = int(input("Enter the required size: "))
        size2 = int(input("Enter the required size2: "))
    # for size in sizes:
        data = [[255 for i in range(size)] for ii in range(size2)]
        # data = [[255 for i in range(size)] for ii in range(size)]
        # data = [[255 for i in range(852)] for ii in range(861)]

        array = np.array(data, dtype=np.uint8)
        image = im.fromarray(array)
        image.save("output_image.png")

        for sequence in sequence_m:
            image = cv.imread("output_image.png")

            # m_factor = size / original_size

            # prev = [0, 0]
            prev = None
            # prev_point_ = [0, 0]
            prev_point_ = None

            # sequence = sequence[:len(sequence) - 1]

            for point in range(len(sequence)):
                curr = sequence[point][0]

                try:
                    point_ = [int(round(size2 * (curr[0] / original_size[0]))), int(round(size * (curr[1] / original_size[1])))]
                    # print("point_:", point_)

                    if prev_point_:
                        if np.linalg.norm(np.array(prev_point_) - np.array(point_)) < 42:
                        # if True:
                            image = cv.line(image, prev_point_, point_, (0, 0, 0), 1)
                            # image = cv.circle(image, point_, radius=0, color=(0, 0, 255), thickness=-1)
                        # image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

                    prev_point_ = point_

                    # prev = curr
                    #
                    # original_distance = np.sqrt(((prev[0] - curr[0]) ** 2) + ((prev[1] - curr[1]) ** 2))
                    #
                    # new_distance = original_distance * m_factor / 2
                    #
                    # if sequence[point][1]:
                    #
                    #     point_ = find_integral_coordinates(prev_point_[0], prev_point_[1], new_distance, sequence[point][1])
                    #     print("point_:", point_)
                    #     print("data HapticProjectConverter:", [original_distance, new_distance], [prev_point_, point_], [prev, curr])
                    #
                    #     if prev:
                    #         image = cv.line(image, prev_point_, point_, (0, 0, 0), 1)
                    #         # image = cv.line(image, prev, curr, (0, 0, 0), 1)
                    #     prev_point_ = point_
                    #
                    # else:
                    #     if curr:
                    #         image = cv.line(image, prev, curr, (0, 0, 0), 1)
                    #
                    # print("starting_point:", starting_point)
                    #
                    # data[point_[1]][point_[0]] = 0

                except Exception as e:
                    print("error at point_:", curr, e)

                prev = curr

            cv.imwrite("output_image.png", image)

def func(original_size, original_beta, original_point):
    pass
    # size = int(input("Enter the required size: "))
    #
    # data = [[255 for i in range(size)] for ii in range(size)]
    #
    # m_factor = size / original_size
    # original_distance = np.sqrt(((prev[0] - original_point[0]) ** 2) + ((prev[1] - original_point[1]) ** 2))
    #
    # new_distance = original_distance * m_factor
    #
    # starting_point = find_integral_coordinates(prev[0], prev[1], new_distance, original_beta)
    # print("starting_point:", starting_point)
    #
    # array = np.array(data, dtype=np.uint8)
    # image = im.fromarray(array)
    # image.save("output_image.png")

    # image.show()

# # Example 2D array (replace this with your own data)
# data = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
# [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31],
# [32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47],
# [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63],
# [64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79],
# [80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95],
# [96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111],
# [112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127],
# [128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143],
# [144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159],
# [160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175],
# [176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191],
# [192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207],
# [208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223],
# [224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239],
# [240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255]]
