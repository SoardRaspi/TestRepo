import sys

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from collections import defaultdict

from scipy.spatial import distance
from sklearn.cluster import DBSCAN

from timeit import default_timer as timer
from colorama import Fore, Style

start = timer()

# D2I = None
dir_dis = {"N": np.array([0, -1]), "NE": np.array([+1, -1]), "E": np.array([+1, 0]), "SE": np.array([+1, +1]),
           "S": np.array([0, +1]), "SW": np.array([-1, +1]), "W": np.array([-1, 0]), "NW": np.array([-1, -1])}
dict_bool = {0: "N", 1: "NE", 2: "E", 3: "SE", 4: "S", 5: "SW", 6: "W", 7: "NW"}
dirs = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
opp = {"N": "S", "NE": "SW", "E": "W", "SE": "NW", "S": "N", "SW": "NE", "W": "E", "NW": "SE"}
opp2 = {"N": ["S", "SW", "SE"], "NE": ["W", "SW", "S"], "E": ["W", "NW", "SW"], "SE": ["N", "W", "NW"],
        "S": ["N", "NE", "NW"], "SW": ["N", "E", "NE"], "W": ["NE", "E", "SE"], "NW": ["S", "E", "SE"]}
mid = {"NE": ["W", "N"], "NW": ["E", "N"], "SE": ["W", "S"], "SW": ["E", "S"]}
fc_nxt = {"NE": ["E", "N"], "NW": ["W", "N"], "SE": ["E", "S"], "SW": ["W", "S"],
          "E": ["NE", "SE"], "S": ["SE", "SW"], "W": ["SW", "NW"], "N": ["NW", "NE"]}
perp = {"N": "E", "NE": "SE", "E": "S", "SE": "SW", "S": "E", "SW": "SE", "W": "S", "NW": "SW"}
basic = ["N", "E", "S", "W"]
juncs = {}

ANS = {}

resolution = 1  # decides the resolution with which the hand movements will happen unless interrupted by direction change
intensity_threshold = 127
ans = ""

centers_traversing = {}

image = cv.imread('test image.png')
# image = cv.imread('rectangle.png')
# image = cv.imread('scythe.png')

avgBlur_thinner = cv.blur(image, (5, 5))
image_final_thinner = cv.cvtColor(avgBlur_thinner, cv.COLOR_BGR2GRAY)
image_final_thinner = np.array(image_final_thinner)
# cv.imshow("thinner", avgBlur_thinner)

avgBlur = cv.medianBlur(image, 5)
# avgBlur = cv.bilateralFilter(image, 7, 75, 75)

# cv.imshow("BLur:", avgBlur)

image_final_t = cv.cvtColor(avgBlur, cv.COLOR_BGR2GRAY)
image_final_test = np.bitwise_not(image_final_t)
image_final_test = np.float32(image_final_test)
image_final = np.array(image_final_t)

maintainer_matrix = [[255 for i in range(len(image_final[0]))] for ii in range(len(image_final))]

# start1 = timer()


def nearest_locator(IMG_F, intensity):
    r = 0
    flag = False
    flag_temp = False
    list_coords = []

    # plt.figure("image_final early:")
    # plt.imshow(IMG_F)

    if IMG_F[0][0] <= intensity:
        flag = True
        list_coords = [0, 0]  # [X, Y]

    while flag == False:
        r += 1
        list_coords = []

        list_coords_h = [[x, r] for x in range(r + 1)]
        list_coords_v = [[r, y] for y in range(r + 1)]
        list_coords_t = []

        for i in range(r + 1):
            list_coords_t.append(list_coords_h[i])
            list_coords_t.append(list_coords_v[i])

        del list_coords_t[-1]

        dist = np.sqrt((list_coords_t[-1][0] ** 2) + (list_coords_t[-1][1] ** 2))

        for coord in list_coords_t:
            if IMG_F[coord[1]][coord[0]] <= intensity:
                dist_temp = np.sqrt((coord[0] ** 2) + (coord[1] ** 2))

                if dist_temp < dist:
                    dist = dist_temp
                    list_coords = [coord]
                    if flag_temp == False:
                        flag_temp = True
                    else:
                        flag = True

                elif dist_temp == dist:
                    list_coords.append(coord)

    print("list_coords:", list_coords)
    return list_coords[0]


XO, YO = nearest_locator(image_final, 75)
print("[X, Y] from nearest function:", [XO, YO])

# X, Y = 118, 76
# X, Y = 201, 129
# X, Y = 96, 99
# X, Y = 289, 146

dst = cv.cornerHarris(image_final_test, 2, 13,
                      0.05)  # image, block_size, kernel_size, k (Harris detector free parameter)

dst = cv.dilate(dst, None)
image_Harris = image.copy()
image_Harris[dst > 0.01 * dst.max()] = [0, 0, 255]

# image_final_Harris = image_final.copy()
image_final_Harris = np.zeros([len(image_final), len(image_final[0])])

coords = [[x, y] for y in range(len(image_Harris)) for x in range(len(image_Harris[0])) if
          (image_Harris[y][x][2] == 255) and (image_Harris[y][x][1] == 0) and (image_Harris[y][x][0] == 0)]
coords_final = [coord for coord in coords if image_final_Harris[coord[1]][coord[0]] < 200]

coords_for_groups = {}
colors_for_groups = {}


def HarrisCorners(COORDS_FINAL, IMAGE_FINAL):
    global coords_for_groups
    global colors_for_groups

    eps = 3

    distances = distance.cdist(COORDS_FINAL, COORDS_FINAL, 'euclidean')

    db = DBSCAN(eps=eps, min_samples=1, metric='precomputed').fit(distances)

    labels = db.labels_

    count = 0

    # Print the groups of close coordinates
    for label in set(labels):
        count += 1

        group = [coord for i, coord in enumerate(COORDS_FINAL) if labels[i] == label]

        colors_for_groups[label] = int(100 + count)
        # # colors_for_groups[300 + count] = label

        coords_for_groups[label] = group

        # flag_lies_on_figure = 1
        # for coord in group:
        #     if flag_lies_on_figure == 1:
        #         # if image_final[coord[1]][coord[0]] > intensity_threshold:
        #         if image_final_t[coord[1]][coord[0]] > 17:
        #             flag_lies_on_figure = 0
        #
        # if flag_lies_on_figure == 1:
        #     colors_for_groups[label] = int(100 + count)
        #     # colors_for_groups[300 + count] = label
        #
        #     coords_for_groups[label] = group

    for label in coords_for_groups.keys():
        coords_corners = coords_for_groups[label]

        for coord_corner in coords_corners:
            IMAGE_FINAL[coord_corner[1]][coord_corner[0]] = colors_for_groups[label]
            maintainer_matrix[coord_corner[1]][coord_corner[0]] = "h" + str(label)

    colors_for_groups = {v: k for k, v in colors_for_groups.items()}

    # plt.figure("IMAGE_FINAL from HARRIS")
    # plt.imshow(IMAGE_FINAL)

    # print("special:", IMAGE_FINAL[256][322])

    print("colors_for_groups:", colors_for_groups)
    print("coords_for_groups:", coords_for_groups)

    return colors_for_groups, coords_for_groups


dict_for_colors, dict_for_coords = HarrisCorners(coords_final, image_final_Harris)

coords_for_COI = {}

for color in dict_for_colors.keys():
    coords_Harris = dict_for_coords[dict_for_colors[color]]
    # coords_Harris_intensities = [image_final_Harris[coord[1]][coord[0]] for coord in coords_Harris]
    coords_Harris_intensities = [image_final_thinner[coord[1]][coord[0]] for coord in coords_Harris]
    Harris_sum_intensities = sum(coords_Harris_intensities)
    Harris_origin = coords_Harris[0]
    coords_Harris_rel = [[coord[0] - Harris_origin[0], coord[1] - Harris_origin[1]] for coord in coords_Harris]

    # print(Harris_sum_intensities, coords_Harris_rel)

    SUM_X = 0
    for i in range(len(coords_Harris_rel)):
        SUM_X += coords_Harris_rel[i][0] * coords_Harris_intensities[i]

    SUM_Y = 0
    for i in range(len(coords_Harris_rel)):
        SUM_Y += coords_Harris_rel[i][1] * coords_Harris_intensities[i]

    # print("SUM_X, SUM_Y:", SUM_X, SUM_Y)

    Harris_COI_X = np.floor(SUM_X / Harris_sum_intensities)
    Harris_COI_Y = np.floor(SUM_Y / Harris_sum_intensities)

    # print("coords_Harris_rel:", coords_Harris_rel)
    # print("coords_Harris_intensities:", coords_Harris_intensities)
    # print("Harris_COI_X, Harris_COI_Y:", Harris_COI_X, Harris_COI_Y)

    # diff_x = int(Harris_COI_X - CX)
    # diff_y = int(Harris_COI_Y - CY)

    CX_Temp, CY_Temp = Harris_origin[0] + Harris_COI_X, Harris_origin[1] + Harris_COI_Y
    coords_for_COI[dict_for_colors[color]] = [int(CX_Temp), int(CY_Temp)]

print("coords_for_COI:", coords_for_COI)

for label in coords_for_COI:
    maintainer_matrix[coords_for_COI[label][1]][coords_for_COI[label][0]] = "h" + str(label)


def group_numbers(NUMBERS, MAX_DIFFERENCE):
    groups = []
    for number in NUMBERS:
        found_group = False
        for group in groups:
            for member in group:
                if abs(member - number) <= MAX_DIFFERENCE:
                    group.append(number)
                    found_group = True

                    # print("print at 38")
                    break

                if found_group:
                    # print("print at 43")
                    break
        if not found_group:
            groups.append([number])

    return groups


def kernel_operator(dir, Y, X, step):
    yy, xx = 0, 0

    if dir == "N":
        yy = Y - step
        xx = X
    elif dir == "NE":
        yy = Y - step
        xx = X + step
    elif dir == "E":
        yy = Y
        xx = X + step
    elif dir == "SE":
        yy = Y + step
        xx = X + step
    elif dir == "S":
        yy = Y + step
        xx = X
    elif dir == "SW":
        yy = Y + step
        xx = X - step
    elif dir == "W":
        yy = Y
        xx = X - step
    elif dir == "NW":
        yy = Y - step
        xx = X - step

    print("data form kernel operator:", [X, Y], [xx, yy])
    return yy, xx


# ----------------------------USEFUL FUNCTIONS START----------------------------

# def kernel_determiner(I, CX, CY, UB, LB):
#     KERNEL2 = np.array([[I[CY - 5][CX - 5], I[CY - 5][CX - 4], I[CY - 5][CX - 3], I[CY - 5][CX - 2], I[CY - 5][CX - 1],
#                          I[CY - 5][CX], I[CY - 5][CX + 1], I[CY - 5][CX + 2], I[CY - 5][CX + 3], I[CY - 5][CX + 4],
#                          I[CY - 5][CX + 5]],
#                         [I[CY - 4][CX - 5], I[CY - 4][CX - 4], I[CY - 4][CX - 3], I[CY - 4][CX - 2], I[CY - 4][CX - 1],
#                          I[CY - 4][CX], I[CY - 4][CX + 1], I[CY - 4][CX + 2], I[CY - 4][CX + 3], I[CY - 4][CX + 4],
#                          I[CY - 4][CX + 5]],
#                         [I[CY - 3][CX - 5], I[CY - 3][CX - 4], I[CY - 3][CX - 3], I[CY - 3][CX - 2], I[CY - 3][CX - 1],
#                          I[CY - 3][CX], I[CY - 3][CX + 1], I[CY - 3][CX + 2], I[CY - 3][CX + 3], I[CY - 3][CX + 4],
#                          I[CY - 3][CX + 5]],
#                         [I[CY - 2][CX - 5], I[CY - 2][CX - 4], I[CY - 2][CX - 3], I[CY - 2][CX - 2], I[CY - 2][CX - 1],
#                          I[CY - 2][CX], I[CY - 2][CX + 1], I[CY - 2][CX + 2], I[CY - 2][CX + 3], I[CY - 2][CX + 4],
#                          I[CY - 2][CX + 5]],
#                         [I[CY - 1][CX - 5], I[CY - 1][CX - 4], I[CY - 1][CX - 3], I[CY - 1][CX - 2], I[CY - 1][CX - 1],
#                          I[CY - 1][CX], I[CY - 1][CX + 1], I[CY - 1][CX + 2], I[CY - 1][CX + 3], I[CY - 1][CX + 4],
#                          I[CY - 1][CX + 5]],
#                         [I[CY][CX - 5], I[CY][CX - 4], I[CY][CX - 3], I[CY][CX - 2], I[CY][CX - 1], I[CY][CX],
#                          I[CY][CX + 1], I[CY][CX + 2], I[CY][CX + 3], I[CY][CX + 4], I[CY][CX + 5]],
#                         [I[CY + 1][CX - 5], I[CY + 1][CX - 4], I[CY + 1][CX - 3], I[CY + 1][CX - 2], I[CY + 1][CX - 1],
#                          I[CY + 1][CX], I[CY + 1][CX + 1], I[CY + 1][CX + 2], I[CY + 1][CX + 3], I[CY + 1][CX + 4],
#                          I[CY + 1][CX + 5]],
#                         [I[CY + 2][CX - 5], I[CY + 2][CX - 4], I[CY + 2][CX - 3], I[CY + 2][CX - 2], I[CY + 2][CX - 1],
#                          I[CY + 2][CX], I[CY + 2][CX + 1], I[CY + 2][CX + 2], I[CY + 2][CX + 3], I[CY + 2][CX + 4],
#                          I[CY + 2][CX + 5]],
#                         [I[CY + 3][CX - 5], I[CY + 3][CX - 4], I[CY + 3][CX - 3], I[CY + 3][CX - 2], I[CY + 3][CX - 1],
#                          I[CY + 3][CX], I[CY + 3][CX + 1], I[CY + 3][CX + 2], I[CY + 3][CX + 3], I[CY + 3][CX + 4],
#                          I[CY + 3][CX + 5]],
#                         [I[CY + 4][CX - 5], I[CY + 4][CX - 4], I[CY + 4][CX - 3], I[CY + 4][CX - 2], I[CY + 4][CX - 1],
#                          I[CY + 4][CX], I[CY + 4][CX + 1], I[CY + 4][CX + 2], I[CY + 4][CX + 3], I[CY + 4][CX + 4],
#                          I[CY + 4][CX + 5]],
#                         [I[CY + 5][CX - 5], I[CY + 5][CX - 4], I[CY + 5][CX - 3], I[CY + 5][CX - 2], I[CY + 5][CX - 1],
#                          I[CY + 5][CX], I[CY + 5][CX + 1], I[CY + 5][CX + 2], I[CY + 5][CX + 3], I[CY + 5][CX + 4],
#                          I[CY + 5][CX + 5]]])
#
#     dirs_or2 = {
#         "N": [[0, 3], [0, 4], [0, 5], [0, 6], [0, 7], [1, 4], [1, 5], [1, 6], [2, 4], [2, 5], [2, 6], [3, 4], [3, 5],
#               [3, 6], [4, 5]],
#         "NE": [[4, 6], [3, 7], [2, 7], [2, 8], [3, 8], [1, 7], [1, 8], [1, 9], [2, 9], [3, 9], [0, 8], [0, 9], [0, 10],
#                [1, 10], [2, 10]],
#         "E": [[5, 6], [4, 7], [5, 7], [6, 7], [4, 8], [5, 8], [6, 8], [4, 9], [5, 9], [6, 9], [3, 10], [4, 10], [5, 10],
#               [6, 10], [7, 10]],
#         "SE": [[6, 6], [7, 7], [8, 7], [8, 8], [7, 8], [9, 7], [9, 8], [9, 9], [8, 9], [7, 9], [10, 8], [10, 9],
#                [10, 10], [9, 10], [8, 10]],
#         "S": [[6, 5], [7, 4], [7, 5], [7, 6], [8, 4], [8, 5], [8, 6], [9, 4], [9, 5], [9, 6], [10, 3], [10, 4], [10, 5],
#               [10, 6], [10, 7]],
#         "SW": [[6, 4], [7, 3], [7, 2], [8, 3], [8, 3], [7, 1], [8, 1], [9, 1], [9, 2], [9, 3], [8, 0], [9, 0], [10, 0],
#                [10, 1], [10, 2]],
#         "W": [[5, 4], [4, 3], [5, 3], [6, 3], [4, 2], [5, 2], [6, 2], [4, 1], [5, 1], [6, 1], [3, 0], [4, 0], [5, 0],
#               [6, 0], [7, 0]],
#         "NW": [[4, 4], [3, 3], [3, 2], [2, 2], [2, 3], [3, 1], [2, 1], [1, 1], [1, 2], [1, 3], [2, 0], [1, 0], [0, 0],
#                [0, 1], [0, 2]]}
#
#     probs2 = np.array([np.average([255 - KERNEL2[coords2[0]][coords2[1]] for coords2 in dirs_or2[d2]]) / 255 for d2 in
#                        dirs_or2.keys()])
#     # print("probs2:", probs2)
#
#     # probs = np.array([np.sqrt(np.mean([KERNEL[coords[0]][coords[1]] ** 2 for coords in dirs_or[d]])) / 255 for d in dirs_or.keys()])
#     # print(probs)
#
#     # print(np.var(probs))
#     # print(np.sqrt(np.mean(probs ** 2)))
#
#     # ans2 = [dirs[index] for index in range(8) if probs[index] < THRESH]
#     ans2 = [dirs[index2] for index2 in range(8) if (probs2[index2] < UB) and (probs2[index2] > LB)]
#     # print("ans2:", ans2)
#
#     probs2_final = group_numbers(probs2, 0.2)[-1]
#     ans2_final = [dirs[index] for index in range(8) if probs2[index] in probs2_final]
#
#     # TODO: To later uncomment for debugging # print("ans2, probs2 from kernel_determiner:", [CX, CY], ans2, probs2)
#
#     # print("probs2:", probs2)
#     # print("grouping:", group_numbers(probs2, 0.2))
#
#     # return ans
#     # return ans2, probs2
#     # print("ans2_final, probs2, probs2_final:", ans2_final, probs2, probs2_final, [CX, CY])
#
#     return ans2_final, probs2_final, ans2
#
#
# def kernel_determiner_bigger(I, CX, CY, UB, LB):
#     KERNEL2 = [[[], [], [], [], [], [], [], [], [], [], [], [], []],
#                [[], [], [], [], [], [], [], [], [], [], [], [], []],
#                [[], [], [], [], [], [], [], [], [], [], [], [], []],
#                [[], [], [], [], [], [], [], [], [], [], [], [], []],
#                [[], [], [], [], [], [], [], [], [], [], [], [], []],
#                [[], [], [], [], [], [], [], [], [], [], [], [], []],
#                [[], [], [], [], [], [], [], [], [], [], [], [], []],
#                [[], [], [], [], [], [], [], [], [], [], [], [], []],
#                [[], [], [], [], [], [], [], [], [], [], [], [], []],
#                [[], [], [], [], [], [], [], [], [], [], [], [], []],
#                [[], [], [], [], [], [], [], [], [], [], [], [], []],
#                [[], [], [], [], [], [], [], [], [], [], [], [], []],
#                [[], [], [], [], [], [], [], [], [], [], [], [], []]]
#
#     for y in range(13):
#         for x in range(13):
#             KERNEL2[y][x] = I[CY - 6 + y][CX - 6 + x]
#
#     # print("KERNEL2 from bigger:", KERNEL2)
#
#     dirs_or2 = {
#         "N": [[0, 3], [0, 4], [0, 5], [0, 6], [0, 7], [0, 8], [0, 9], [1, 4], [1, 5], [1, 6], [1, 7], [1, 8], [2, 4],
#               [2, 5], [2, 6], [2, 7], [2, 8], [3, 5], [3, 6], [3, 7], [4, 5], [4, 6], [4, 7], [5, 6]],
#         "NE": [[0, 9], [0, 10], [0, 11], [0, 12], [1, 12], [2, 12], [3, 12], [1, 8], [1, 9], [1, 10], [1, 11], [2, 11],
#                [3, 11], [4, 11], [2, 8], [2, 9], [2, 10], [3, 10], [4, 10], [3, 7], [3, 8], [3, 9], [4, 9], [5, 9],
#                [4, 7], [4, 8], [5, 8], [5, 7]],
#         "E": [[6, 7], [5, 8], [6, 8], [7, 8], [5, 9], [6, 9], [7, 9], [4, 10], [5, 10], [6, 10], [7, 10], [8, 10],
#               [4, 11], [5, 11], [6, 11], [7, 11], [8, 11], [3, 12], [4, 12], [5, 12], [6, 12], [7, 12], [8, 12],
#               [9, 12]],
#         "SE": [[7, 7], [8, 7], [8, 8], [7, 8], [9, 7], [9, 8], [9, 9], [8, 9], [7, 9], [10, 8], [10, 9], [10, 10],
#                [9, 10], [8, 10], [11, 8], [11, 9], [11, 10], [11, 11], [10, 11], [9, 11], [8, 11], [12, 9], [12, 10],
#                [12, 11], [12, 12], [11, 12], [10, 12], [9, 12]],
#         "S": [[7, 6], [8, 5], [8, 6], [8, 7], [9, 5], [9, 6], [9, 7], [10, 4], [10, 5], [10, 6], [10, 7], [10, 8],
#               [11, 4], [11, 5], [11, 6], [11, 7], [11, 8], [12, 3], [12, 4], [12, 5], [12, 6], [12, 7], [12, 8],
#               [12, 9]],
#         "SW": [[7, 5], [7, 4], [8, 4], [8, 5], [7, 3], [8, 3], [9, 3], [9, 4], [9, 5], [8, 2], [9, 2], [10, 2], [10, 3],
#                [10, 4], [8, 1], [9, 1], [10, 1], [11, 1], [11, 2], [11, 3], [11, 4], [9, 0], [10, 0], [11, 0], [12, 0],
#                [12, 1], [12, 2], [12, 3]],
#         "W": [[6, 5], [5, 4], [6, 4], [7, 4], [5, 3], [6, 3], [7, 3], [4, 2], [5, 2], [6, 2], [7, 2], [8, 2], [4, 1],
#               [5, 1], [6, 1], [7, 1], [8, 1], [3, 0], [4, 0], [5, 0], [6, 0], [7, 0], [8, 0], [9, 0]],
#         "NW": [[5, 5], [5, 4], [4, 4], [4, 5], [5, 3], [4, 3], [3, 3], [3, 4], [3, 5], [4, 2], [3, 2], [2, 2], [2, 3],
#                [2, 4], [4, 1], [3, 1], [2, 1], [1, 1], [1, 2], [1, 3], [1, 4], [3, 0], [2, 0], [1, 0], [0, 0], [0, 1],
#                [0, 2], [0, 3]]}
#
#     probs2 = np.array(
#         [np.average([KERNEL2[coords2[0]][coords2[1]] for coords2 in dirs_or2[d2]]) / 255 for d2 in dirs_or2.keys()])
#     # print(probs2)
#
#     # probs = np.array([np.sqrt(np.mean([KERNEL[coords[0]][coords[1]] ** 2 for coords in dirs_or[d]])) / 255 for d in dirs_or.keys()])
#     # print(probs)
#
#     # print(np.var(probs))
#     # print(np.sqrt(np.mean(probs ** 2)))
#
#     # ans2 = [dirs[index] for index in range(8) if probs[index] < THRESH]
#     ans2 = [dirs[index2] for index2 in range(8) if (probs2[index2] < UB) and (probs2[index2] > LB)]
#
#     # probs2_final = group_numbers(probs2, 0.2)[-1]
#     probs2_final = probs2.tolist()
#
#     ans2_final = [dirs[index] for index in range(8) if probs2[index] in probs2_final]
#
#     # TODO: To later uncomment for debugging # print("ans2, probs2 from kernel_determiner_bigger:", [CX, CY], ans2, probs2)
#
#     # print("probs2:", probs2)
#     # print("grouping:", group_numbers(probs2, 0.2))
#
#     # return ans
#     # return ans2, probs2
#     return ans2_final, probs2_final

# ----------------------------USEFUL FUNCTIONS END----------------------------


# def find_pyramidal_conical_structures(X, Y, Z, threshold=0.5):
#     """
#     Find pyramidal or conical structures in a 3D plot.
#
#     Args:
#         X, Y, Z (numpy arrays): The X, Y, and Z coordinates of the points in the 3D plot.
#         threshold (float, optional): The threshold value to determine the height of the structure. Default is 0.5.
#
#     Returns:
#         List of 3D polygon collections representing the pyramidal or conical structures.
#     """
#     structures = []
#     coords = []
#
#     count = 0
#
#     for i in range(X.shape[0] - 1):
#         for j in range(X.shape[1] - 1):
#             if Z[i, j] > threshold and Z[i + 1, j] > threshold and Z[i, j + 1] > threshold and Z[
#                 i + 1, j + 1] > threshold:
#                 vertices = np.array([(X[i, j], Y[i, j], Z[i, j]),
#                                      (X[i + 1, j], Y[i + 1, j], Z[i + 1, j]),
#                                      (X[i + 2, j], Y[i + 2, j], Z[i + 2, j]),
#                                      (X[i + 1, j + 1], Y[i + 1, j + 1], Z[i + 1, j + 1]),
#                                      (X[i + 2, j + 1], Y[i + 2, j + 1], Z[i + 2, j + 1]),
#                                      (X[i + 2, j + 2], Y[i + 2, j + 2], Z[i + 2, j + 2]),
#                                      (X[i + 1, j + 2], Y[i + 1, j + 2], Z[i + 1, j + 2]),
#                                      (X[i, j + 1], Y[i, j + 1], Z[i, j + 1]),
#                                      (X[i, j + 2], Y[i, j + 2], Z[i, j + 2])])
#                 poly = Poly3DCollection([vertices], alpha=0.5, edgecolor='black')
#                 structures.append(poly)
#
#                 # plt.figure(count)
#                 # print("count:", count, [j, i])
#                 # plt.imshow(vertices)
#
#                 # plt.show()
#                 # count += 1
#
#                 coords.append(vertices)
#
#     return structures, coords


# def slope(CX, CY, DIRECTION, IM):
#     point_y, point_x = kernel_operator(DIRECTION, CY, CX, 1)
#
#     delta_x = point_x - CX
#     delta_y = point_y - CY
#
#     # IM = IM.astype(float)
#
#     # delta_z = (IM[point_y][point_x] - IM[CY][CX]) / 255
#     # delta_z = IM[point_y][point_x] - IM[CY][CX]
#
#     # distance = np.sqrt(delta_x**2 + delta_y**2 + delta_z**2)
#     distance = np.sqrt(delta_x ** 2 + delta_y ** 2)
#
#     # print("----------")
#     # # print("delta_z, distance:", delta_z, distance, [CX, CY])
#     # print("intensities:", (IM[CY][CX] - IM[point_y][point_x]) / distance)
#     # print("----------")
#
#     return (IM[CY][CX].astype(float) - IM[point_y][point_x].astype(float)) / distance
#
#     # return (delta_x / distance, delta_y / distance, delta_z / distance)
#     # return delta_z / distance


centers = []


def grapher_3D(SIZE, IMG_F, CX, CY):
    start_grapher = timer()
    # CX, CY = 320, 505
    # CX, CY = 499, 253
    # CX, CY = 452, 503
    # CX, CY = 700, 754
    # CX, CY = 452, 506
    # CX, CY = 202, 382
    # CX, CY = 500, 251
    # CX, CY = 240, 168
    # CX, CY = 201, 129
    # CX, CY = 211, 141
    # CX, CY = 250, 178

    # CX, CY = 497, 253
    # CX, CY = 322, 253
    # CX, CY = 203, 380
    # CX, CY = 320, 505
    # CX, CY = 453, 505
    # CX, CY = 700, 505
    # CX, CY = 454, 751
    # CX, CY = 700, 751
    # CX, CY = 100, 725
    # CX, CY = 200, 129

    # CX, CY = 320, 504

    data_2d = np.zeros([SIZE, SIZE])
    values = []
    values_otherwise = []

    max_point = 0

    mid = (SIZE - 1) // 2

    corner_flag = 0
    corner_color = 0

    for y in range(SIZE):
        for x in range(SIZE):
            # temp_inner = 255 * np.sin((255 - IMG_F[CY - mid + y][CX - mid + x]) / 255)
            temp_inner = 255 - IMG_F[CY - mid + y][CX - mid + x]
            # data_2d[y][x] = np.sqrt((255 - IMG_F[CY - mid + y][CX - mid + x])**2 + (255 - IMG_F[CY - mid + y + 1][CX - mid + x])**2 + (255 - IMG_F[CY - mid + y - 1][CX - mid + x])**2)
            data_2d[y][x] = temp_inner

            # print("temp_inner in grapher_3D:", temp_inner)

            # if temp_inner > max_value:
            #     max_value = temp_inner
            #     center_x, center_y = x, y

            if temp_inner > intensity_threshold:
                # if temp_inner > 120:
                values.append(temp_inner)
            else:
                values_otherwise.append(temp_inner)

            # if temp_inner > max_point:
            #     max_point = temp_inner

            if corner_flag == 0:
                temp_inner = image_final_Harris[CY - mid + y][CX - mid + x]
                # print("temp_inner from image_final_Harris:", corner_flag, temp_inner, [CX - mid + x, CY - mid + y],
                #       [CX, CY])

                if temp_inner != 0:
                    # corner_color = int(255 - temp_inner)
                    corner_color = temp_inner

                    corner_flag = 1

    # diff_x, diff_y = 0, 0

    # TODO: Uncomment 421-430 to re-center to the maximum pixel intensity

    diff_x, diff_y = 0, 0

    # print("corner_flag:", corner_flag)
    # print("corner_color:", corner_color)
    # print("image_final_Harris:", image_final_Harris)

    # plt.figure("image_final_Harris")
    # plt.imshow(image_final_Harris)

    data_2d_corner = 255 * np.ones([SIZE, SIZE])

    if corner_flag == 1:
        coords_Harris = dict_for_coords[dict_for_colors[corner_color]]
        # coords_Harris_intensities = [image_final_Harris[coord[1]][coord[0]] for coord in coords_Harris]
        coords_Harris_intensities = [image_final_thinner[coord[1]][coord[0]] for coord in coords_Harris]
        Harris_sum_intensities = sum(coords_Harris_intensities)
        Harris_origin = coords_Harris[0]
        coords_Harris_rel = [[coord[0] - Harris_origin[0], coord[1] - Harris_origin[1]] for coord in coords_Harris]

        # print(Harris_sum_intensities, coords_Harris_rel)

        SUM_X = 0
        for i in range(len(coords_Harris_rel)):
            SUM_X += coords_Harris_rel[i][0] * coords_Harris_intensities[i]

        SUM_Y = 0
        for i in range(len(coords_Harris_rel)):
            SUM_Y += coords_Harris_rel[i][1] * coords_Harris_intensities[i]

        # print("SUM_X, SUM_Y:", SUM_X, SUM_Y)

        Harris_COI_X = np.floor(SUM_X / Harris_sum_intensities)
        Harris_COI_Y = np.floor(SUM_Y / Harris_sum_intensities)

        # print("coords_Harris_rel:", coords_Harris_rel)
        # print("coords_Harris_intensities:", coords_Harris_intensities)
        # print("Harris_COI_X, Harris_COI_Y:", Harris_COI_X, Harris_COI_Y)

        # diff_x = int(Harris_COI_X - CX)
        # diff_y = int(Harris_COI_Y - CY)

        CX_Temp, CY_Temp = Harris_origin[0] + Harris_COI_X, Harris_origin[1] + Harris_COI_Y

        # print("CX_Temp, CY_Temp:", CX_Temp, CY_Temp)

        diff_x = int(CX_Temp - CX)
        diff_y = int(CY_Temp - CY)

        for i in range(len(coords_Harris_rel)):
            coord_rel = coords_Harris_rel[i]

            # data_2d_corner[coord_rel[0] + 5][coord_rel[1] + 5] = coords_Harris_intensities[i]
            data_2d_corner[coord_rel[0] + 2][coord_rel[1] + 2] = coords_Harris_intensities[i]

    for y in range(SIZE):
        for x in range(SIZE):
            data_2d_corner[y][x] = 255 - data_2d_corner[y][x]

    values = [*set(values)]
    values.sort(reverse=True)

    values_otherwise = [*set(values)]
    values_otherwise.sort(reverse=True)

    print("values_otherwise before del values[0]:", values_otherwise)
    print("values before del values[0]:", values, [CX, CY])

    del values[0]

    if [CX + diff_x, CY + diff_y] not in centers:
        centers.append([CX + diff_x, CY + diff_y])

    # print("[CX, CY]:", [CX + diff_x, CY + diff_y], centers)
    # # plt.show()

    for y in range(SIZE):
        for x in range(SIZE):
            temp_inner = 255 * np.sin((255 - IMG_F[CY + diff_y - mid + y][CX + diff_x - mid + x]) / 255)
            # temp_inner = 255 * np.sin((255 - IMG_F_THINNER[CY + diff_y - mid + y][CX + diff_x - mid + x]) / 255)

            data_2d[y][x] = temp_inner

            if temp_inner > intensity_threshold:
                values.append(temp_inner)

    # X_grid, Y_grid = np.meshgrid(np.arange(CX + diff_x - mid, CX + diff_x + mid + 1, 1), np.arange(CY + diff_y - mid, CY + diff_y + mid + 1, 1))
    X_grid, Y_grid = np.meshgrid(np.arange(CX + diff_x - mid, CX + diff_x + mid + 1, 1),
                                 np.arange(CY + diff_y - mid, CY + diff_y + mid + 1, 1))

    # # Create a 3D figure
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    #
    # plt.contour(X_grid, Y_grid, data_2d, colors='black')
    #
    # # Plot the 2D array as a surface plot
    # # ax.plot_surface(X_grid, Y_grid, data_2d)
    # ax.set_ylim(CY + diff_y + mid + 1, CY + diff_y - mid)
    #
    # # Set labels and title
    # ax.set_xlabel('X Axis Label')
    # ax.set_ylabel('Y Axis Label')
    # ax.set_zlabel('Z Axis Label')
    # # ax.set_title('3D Plot centered at (' + str(CX) + ', ' + str(CY) + ')')
    # ax.set_title('3D Plot centered at (' + str(CX + diff_x) + ', ' + str(CY + diff_y) + ')')

    # prob_dirs, prob_dirs_probs = kernel_determiner_bigger(IMG_F, CX + diff_x, CY + diff_y, 0, 1)
    # print("prob_dirs, prob_dirs_probs:", prob_dirs, prob_dirs_probs)

    # return find_dir_graph_function(values, data_2d), X_grid, Y_grid, data_2d, CX + diff_x, CY + diff_y

    # data_2d_inner = [[0] * len(image_final[0])] * len(image_final)

    # plt.figure("data_2d_inner")
    # # plt.grid()
    # plt.imshow(data_2d_inner)

    end_grapher = timer()
    print("time for grapher_3D:", end_grapher - start_grapher)

    return X_grid, Y_grid, data_2d, CX + diff_x, CY + diff_y


def find_dir_graph_function(VALUES, DATA_2D):
    dirs = {'N': [],
            'NE': [],
            'E': [],
            'SE': [],
            'S': [],
            'SW': [],
            'W': [],
            'NW': []}

    dirs_or2 = {
        "N": [[5, 6], [4, 5], [4, 6], [4, 7], [3, 5], [3, 6], [3, 7], [2, 5], [2, 6], [2, 7], [1, 4], [1, 5], [1, 6],
              [1, 7], [1, 8], [0, 3], [0, 4], [0, 5], [0, 6], [0, 7], [0, 8], [0, 9]],
        "NE": [[5, 7], [2, 8], [3, 8], [4, 8], [1, 9], [2, 9], [3, 9], [4, 9], [0, 10], [1, 10], [2, 10], [3, 10],
               [4, 10], [0, 11], [1, 11], [2, 11], [3, 11], [0, 12], [1, 12], [2, 12]],
        "E": [[6, 7], [5, 8], [6, 8], [7, 8], [5, 9], [6, 9], [7, 9], [5, 10], [6, 10], [7, 10], [4, 11], [5, 11],
              [6, 11], [7, 11], [8, 11], [3, 12], [4, 12], [5, 12], [6, 12], [7, 12], [8, 12], [9, 12]],
        "SE": [[7, 7], [8, 8], [8, 9], [8, 10], [9, 8], [9, 9], [9, 10], [9, 11], [10, 8], [10, 9], [10, 10], [10, 11],
               [10, 12], [11, 9], [11, 10], [11, 11], [11, 12], [12, 10], [12, 11], [12, 12]],
        "S": [[7, 6], [8, 5], [8, 6], [8, 7], [9, 5], [9, 6], [9, 7], [10, 5], [10, 6], [10, 7], [11, 4], [11, 5],
              [11, 6], [11, 7], [11, 8], [12, 3], [12, 4], [12, 5], [12, 6], [12, 7], [12, 8], [12, 9]],
        "SW": [[7, 5], [8, 2], [8, 3], [8, 4], [9, 1], [9, 2], [9, 3], [9, 4], [10, 0], [10, 1], [10, 2], [10, 3],
               [10, 4], [11, 0], [11, 1], [11, 2], [11, 3], [12, 0], [12, 1], [12, 2]],
        "W": [[6, 5], [5, 4], [6, 4], [7, 4], [5, 3], [6, 3], [7, 3], [5, 2], [6, 2], [7, 2], [4, 1], [5, 1], [6, 1],
              [7, 1], [8, 1], [3, 0], [4, 0], [5, 0], [6, 0], [7, 0], [8, 0], [9, 0]],
        "NW": [[5, 5], [4, 2], [4, 3], [4, 4], [3, 1], [3, 2], [3, 3], [3, 4], [2, 0], [2, 1], [2, 2], [2, 3], [2, 4],
               [1, 0], [1, 1], [1, 2], [1, 3], [0, 0], [0, 1], [0, 2]]}

    layers_dirs = {"N": {"l0": [[5, 6]],
                         "l1": [[4, 5], [4, 6], [4, 7]],
                         "l2": [[3, 5], [3, 6], [3, 7]],
                         "l3": [[2, 5], [2, 6], [2, 7]],
                         # "l4": [[1, 4], [1, 5], [1, 6], [1, 7], [1, 8]],
                         "l4": [[1, 5], [1, 6], [1, 7]],
                         # "l5": [[0, 3], [0, 4], [0, 5], [0, 6], [0, 7], [0, 8], [0, 9]]},
                         "l5": [[0, 5], [0, 6], [0, 7]]},
                   "NE": {"l0": [[5, 7]],
                          "l1": [[4, 8]],
                          "l2": [[3, 8], [3, 9], [4, 9]],
                          "l3": [[2, 8], [2, 9], [2, 10], [3, 10], [4, 10]],
                          "l4": [[1, 9], [1, 10], [1, 11], [2, 11], [3, 11]],
                          "l5": [[0, 10], [0, 11], [0, 12], [1, 12], [2, 12]]},
                   "E": {"l0": [[6, 7]],
                         "l1": [[5, 8], [6, 8], [7, 8]],
                         "l2": [[5, 9], [6, 9], [7, 9]],
                         "l3": [[5, 10], [6, 10], [7, 10]],
                         # "l4": [[4, 11], [5, 11], [6, 11], [7, 11], [8, 11]],
                         "l4": [[5, 11], [6, 11], [7, 11]],
                         # "l5": [[3, 12], [4, 12], [5, 12], [6, 12], [7, 12], [8, 12], [9, 12]]},
                         "l5": [[5, 12], [6, 12], [7, 12]]},
                   "SE": {"l0": [[7, 7]],
                          "l1": [[8, 8]],
                          "l2": [[9, 8], [9, 9], [8, 9]],
                          "l3": [[10, 8], [10, 9], [10, 10], [9, 10], [8, 10]],
                          "l4": [[11, 9], [11, 10], [11, 11], [10, 11], [9, 11]],
                          "l5": [[12, 10], [12, 11], [12, 12], [11, 12], [10, 12]]},
                   "S": {"l0": [[7, 6]],
                         "l1": [[8, 5], [8, 6], [8, 7]],
                         "l2": [[9, 5], [9, 6], [9, 7]],
                         "l3": [[10, 5], [10, 6], [10, 7]],
                         # "l4": [[11, 4], [11, 5], [11, 6], [11, 7], [11, 8]],
                         "l4": [[11, 5], [11, 6], [11, 7]],
                         # "l5": [[12, 3], [12, 4], [12, 5], [12, 6], [12, 7], [12, 8], [12, 9]]},
                         "l5": [[12, 5], [12, 6], [12, 7]]},
                   "SW": {"l0": [[7, 5]],
                          "l1": [[8, 4]],
                          "l2": [[8, 3], [9, 3], [9, 4]],
                          "l3": [[8, 2], [9, 2], [10, 2], [10, 3], [10, 4]],
                          "l4": [[9, 1], [10, 1], [11, 1], [11, 2], [11, 3]],
                          "l5": [[10, 0], [11, 0], [12, 0], [12, 1], [12, 2]]},
                   "W": {"l0": [[6, 5]],
                         "l1": [[5, 4], [6, 4], [7, 4]],
                         "l2": [[5, 3], [6, 3], [7, 3]],
                         "l3": [[5, 2], [6, 2], [7, 2]],
                         # "l4": [[4, 1], [5, 1], [6, 1], [7, 1], [8, 1]],
                         "l4": [[5, 1], [6, 1], [7, 1]],
                         # "l5": [[3, 0], [4, 0], [5, 0], [6, 0], [7, 0], [8, 0], [9, 0]]},
                         "l5": [[5, 0], [6, 0], [7, 0]]},
                   "NW": {"l0": [[5, 5]],
                          "l1": [[4, 4]],
                          "l2": [[4, 3], [3, 3], [3, 4]],
                          "l3": [[4, 2], [3, 2], [2, 2], [2, 3], [2, 4]],
                          "l4": [[3, 1], [2, 1], [1, 1], [1, 2], [1, 3]],
                          "l5": [[2, 0], [1, 0], [0, 0], [0, 1], [0, 2]]}}

    val_layers_dirs = {"N": {"l0": [], "l1": [], "l2": [], "l3": [], "l4": [], "l5": []},
                       "NE": {"l0": [], "l1": [], "l2": [], "l3": [], "l4": [], "l5": []},
                       "E": {"l0": [], "l1": [], "l2": [], "l3": [], "l4": [], "l5": []},
                       "SE": {"l0": [], "l1": [], "l2": [], "l3": [], "l4": [], "l5": []},
                       "S": {"l0": [], "l1": [], "l2": [], "l3": [], "l4": [], "l5": []},
                       "SW": {"l0": [], "l1": [], "l2": [], "l3": [], "l4": [], "l5": []},
                       "W": {"l0": [], "l1": [], "l2": [], "l3": [], "l4": [], "l5": []},
                       "NW": {"l0": [], "l1": [], "l2": [], "l3": [], "l4": [], "l5": []}, }

    data_2d_inner = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # , 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

    valid = []

    for dir in layers_dirs:
        for layer in layers_dirs[dir]:
            layer_coords = layers_dirs[dir][layer]
            values = [DATA_2D[coords[0]][coords[1]] for coords in layer_coords]

            val = max(values)

            # if val in VALUES:
            # print("special value:", VALUES[-7])
            if val >= 160:
                val_coord = layer_coords[values.index(val)]
                val_layers_dirs[dir][layer] = val_coord
                data_2d_inner[val_coord[0]][val_coord[1]] = val

    for dir in val_layers_dirs:
        layers = val_layers_dirs[dir]

        count = 0

        for layer in layers:
            # print("layer:", layer)
            if layers[layer]:
                count += 1

        if count == 6:
            valid.append(dir)

    data_2d_inner = np.float32(data_2d_inner)

    data_2d_inner = [[0] * len(image_final[0])] * len(image_final)

    plt.figure("data_2d_inner")
    # plt.grid()
    plt.imshow(data_2d_inner)

    return valid


def getstep(direction, COI_X, COI_Y, coords_for_grups):
    step = 9

    COI_coords = [COI_X, COI_Y]

    print("coords_for_COI from getstep:", coords_for_COI)

    k = None
    for key in coords_for_COI:
        if not k:
            # print("key:", key, coords_for_COI[key], COI_coords)
            if coords_for_COI[key] == COI_coords:
                k = key
                print("k:", k)

    if k:
        coords_corresponding = coords_for_grups[k]
        # coords_corresponding.remove(COI_coords)

        # direction_required = {}
        direction_required = []

        curr = COI_coords

        for i in coords_corresponding:
            delta_x = curr[0] - i[0]
            delta_y = curr[1] - i[1]
            theta_radians = np.arctan2(delta_y, delta_x)

            angle_new_format = theta_radians / np.pi
            # degs.append(theta_radians)

            # if (curr[0] != X) and (prev[0] != X) and (curr[1] != Y) and (prev[1] != Y):
            if True or ((curr[0] != XO) and (i[0] != XO) and (curr[1] != YO) and (i[1] != YO)):
                direction_pointing = None

                if ((angle_new_format >= 0.125) and (angle_new_format <= 0.375)) or \
                        ((angle_new_format >= -0.875) and (angle_new_format <= -0.625)):
                    direction_pointing = 'SE'
                elif ((angle_new_format >= 0.375) and (angle_new_format < 0.625)) or \
                        ((angle_new_format > -0.625) and (angle_new_format < -0.375)):
                    direction_pointing = 'S'
                elif ((angle_new_format >= 0.625) and (angle_new_format <= 0.875)) or \
                        ((angle_new_format >= -0.375) and (angle_new_format <= -0.125)):
                    direction_pointing = 'SW'
                elif ((angle_new_format >= 0) and (angle_new_format < 0.125)) or \
                        ((angle_new_format >= -1) and (angle_new_format < -0.875)) or \
                        ((angle_new_format > 0.875) and (angle_new_format <= 1)) or \
                        ((angle_new_format > -0.125) and (angle_new_format < 0)):
                    # ((angle_new_format < -0.125) and (angle_new_format > 0)):
                    direction_pointing = 'E'

                # print('direction_pointing before:', direction_pointing)

                if direction_pointing == 'SE':
                    if (curr[1] <= YO) or (i[1] <= YO):
                        direction_pointing = 'NW'
                elif direction_pointing == 'S':
                    if (curr[1] <= YO) or (i[1] <= YO):
                        direction_pointing = 'N'
                elif direction_pointing == 'SW':
                    if (curr[1] <= YO) or (i[1] <= YO):
                        direction_pointing = 'NE'
                elif direction_pointing == 'E':
                    if (curr[0] <= XO) or (i[0] <= XO):
                        direction_pointing = 'W'

                # print('direction_pointing after:', direction_pointing)

                # direction_required.append(direction_pointing)
                if (direction_pointing == direction) or (direction_pointing == opp[direction]):
                    direction_required.append(i)
                    # if direction_pointing not in direction_required:
                    #     direction_required[direction_pointing] = [i]
                    # else:
                    #     direction_required[direction_pointing].append(i)

        max_coord = direction_required[0]
        max_dist = 0

        for ii in direction_required:
            dist_ii = np.sqrt((max_coord[0] - ii[0]) ** 2 + (max_coord[1] - ii[1]) ** 2)

            if dist_ii > max_dist:
                max_coord = ii

        del direction_required

        max_coord[1], max_coord[0] = kernel_operator(direction, max_coord[1], max_coord[0], 1)

        # end_getstep = timer()
        # print("coord_required:", max_coord, end_getstep - start_getstep)
        print("coord_required:", max_coord)

        # TODO: -----------------------------------------------------------------------------------------------------------

        # COI_coords = [COI_X, COI_Y]
        #
        # print("coords_for_COI from getstep:", coords_for_COI)
        #
        # k = None
        # for key in coords_for_COI:
        #     if not k:
        #         print("key:", key, coords_for_COI[key], COI_coords)
        #         if coords_for_COI[key] == COI_coords:
        #             k = key
        #             print("k:", k)
        #
        # coords_corresponding = coords_for_grups[k]
        # # coords_corresponding.remove(COI_coords)
        #
        # direction_required = {}
        #
        # curr = COI_coords
        #
        # for i in coords_corresponding:
        #     delta_x = curr[0] - i[0]
        #     delta_y = curr[1] - i[1]
        #     theta_radians = np.arctan2(delta_y, delta_x)
        #
        #     angle_new_format = theta_radians / np.pi
        #     # degs.append(theta_radians)
        #
        #     # if (curr[0] != X) and (prev[0] != X) and (curr[1] != Y) and (prev[1] != Y):
        #     if True or ((curr[0] != X) and (i[0] != X) and (curr[1] != Y) and (i[1] != Y)):
        #         direction_pointing = None
        #
        #         if ((angle_new_format >= 0.125) and (angle_new_format <= 0.375)) or \
        #                 ((angle_new_format >= -0.875) and (angle_new_format <= -0.625)):
        #             direction_pointing = 'SE'
        #         elif ((angle_new_format >= 0.375) and (angle_new_format < 0.625)) or \
        #                 ((angle_new_format > -0.625) and (angle_new_format < -0.375)):
        #             direction_pointing = 'S'
        #         elif ((angle_new_format >= 0.625) and (angle_new_format <= 0.875)) or \
        #                 ((angle_new_format >= -0.375) and (angle_new_format <= -0.125)):
        #             direction_pointing = 'SW'
        #         elif ((angle_new_format >= 0) and (angle_new_format < 0.125)) or \
        #                 ((angle_new_format >= -1) and (angle_new_format < -0.875)) or \
        #                 ((angle_new_format > 0.875) and (angle_new_format <= 1)) or \
        #                 ((angle_new_format > -0.125) and (angle_new_format < 0)):
        # ((angle_new_format < -0.125) and (angle_new_format > 0)):
        #             direction_pointing = 'E'
        #
        #         # print('direction_pointing before:', direction_pointing)
        #
        #         if direction_pointing == 'SE':
        #             if (curr[1] <= Y) or (i[1] <= Y):
        #                 direction_pointing = 'NW'
        #         elif direction_pointing == 'S':
        #             if (curr[1] <= Y) or (i[1] <= Y):
        #                 direction_pointing = 'N'
        #         elif direction_pointing == 'SW':
        #             if (curr[1] <= Y) or (i[1] <= Y):
        #                 direction_pointing = 'NE'
        #         elif direction_pointing == 'E':
        #             if (curr[0] <= X) or (i[0] <= X):
        #                 direction_pointing = 'W'
        #
        #         # print('direction_pointing after:', direction_pointing)
        #
        #         # direction_required.append(direction_pointing)
        #         if direction_pointing not in direction_required:
        #             direction_required[direction_pointing] = [i]
        #         else:
        #             direction_required[direction_pointing].append(i)
        #
        # print("direction_required:", direction_required)

        step = int(np.ceil(np.sqrt((max_coord[0] - COI_coords[0]) ** 2 + (max_coord[1] - COI_coords[1]) ** 2))) + 6
        # print("s:", s, type(s))
    else:
        step = 9

    return step


# def get_dirs_from_coords(_, __, ___, X, Y, PREV):
#     # st_plot = time.time()
#     cs = plt.contour(_, __, ___, colors='black')
#     # end_plot = time.time()
#
#     # print("time for plotting:", st_plot - end_plot)
#
#     flag = 0
#     index_path = 0
#
#     coords = {}
#
#     # print("something:", len(cs.collections))
#
#     while flag != 1:
#         try:
#             p = cs.collections[-3].get_paths()[index_path]
#             # plt.clf()
#             v = p.vertices
#             x = v[:, 0]
#             y = v[:, 1]
#
#             coords[index_path] = [[x[i], y[i]] for i in range(len(x))]
#
#             # plt.scatter(x, y)
#             # print("x, y:", x, y)
#         except:
#             flag = 1
#
#         index_path += 1
#
#     # print("coords_path:", coords, [X, Y])
#
#     # print("[X, Y] from get_dirs_from_coords:", [X, Y])
#
#     # x1, y1 = [X, X], [Y - 6, Y + 6]
#     # x2, y2 = [X - 6, X + 6], [Y, Y]
#     # plt.plot(x1, y1, x2, y2, marker='o')
#
#     final_ans = {}
#     ans = []
#
#     print("len(coords):", len(coords))
#
#     for ii in range(len(coords)):
#         c = coords[ii]
#
#         # coords = COORDS[2]
#         # coords = COORDS[1]
#         # c = coords[0]
#
#         # print("length of COORDS:", len(COORDS))
#
#         # slopes = []
#         degs = []
#         degs_where = []
#         direction = []
#
#         for i in range(1, len(c)):
#             curr = c[i]
#             prev = c[i - 1]
#
#             delta_x = curr[0] - prev[0]
#             delta_y = curr[1] - prev[1]
#             theta_radians = np.arctan2(delta_y, delta_x)
#
#             angle_new_format = theta_radians / np.pi
#             degs.append(theta_radians)
#
#             # if (curr[0] != X) and (prev[0] != X) and (curr[1] != Y) and (prev[1] != Y):
#             if True or ((curr[0] != X) and (prev[0] != X) and (curr[1] != Y) and (prev[1] != Y)):
#                 direction_pointing = None
#
#                 if ((angle_new_format >= 0.125) and (angle_new_format <= 0.375)) or \
#                         ((angle_new_format >= -0.875) and (angle_new_format <= -0.625)):
#                     direction_pointing = 'SE'
#                 elif ((angle_new_format >= 0.375) and (angle_new_format < 0.625)) or \
#                         ((angle_new_format > -0.625) and (angle_new_format < -0.375)):
#                     direction_pointing = 'S'
#                 elif ((angle_new_format >= 0.625) and (angle_new_format <= 0.875)) or \
#                         ((angle_new_format >= -0.375) and (angle_new_format <= -0.125)):
#                     direction_pointing = 'SW'
#                 elif ((angle_new_format >= 0) and (angle_new_format < 0.125)) or \
#                         ((angle_new_format >= -1) and (angle_new_format < -0.875)) or \
#                         ((angle_new_format > 0.875) and (angle_new_format <= 1)) or \
#                         ((angle_new_format > -0.125) and (angle_new_format < 0)):
# ((angle_new_format < -0.125) and (angle_new_format > 0)):
#                     direction_pointing = 'E'
#
#                 # print('direction_pointing before:', direction_pointing)
#
#                 if direction_pointing == 'SE':
#                     if (curr[1] <= Y) or (prev[1] <= Y):
#                         direction_pointing = 'NW'
#                 elif direction_pointing == 'S':
#                     if (curr[1] <= Y) or (prev[1] <= Y):
#                         direction_pointing = 'N'
#                 elif direction_pointing == 'SW':
#                     if (curr[1] <= Y) or (prev[1] <= Y):
#                         direction_pointing = 'NE'
#                 elif direction_pointing == 'E':
#                     if (curr[0] <= X) or (prev[0] <= X):
#                         direction_pointing = 'W'
#
#                 # print('direction_pointing after:', direction_pointing)
#
#                 direction.append(direction_pointing)
#             else:
#                 direction.append(None)
#
#         # print("coords:", coords)
#         # print("direction:", direction)
#
#         temp = [item for item in direction if item]
#
#         final_ans[ii] = list(np.unique(np.array(temp)))
#
#     print("final_ans:", final_ans)
#
#     final_ans['loop'] = final_ans[0]
#
#     for t in range(1, len(coords)):
#         dirs_from_prev = final_ans[t - 1]
#         dirs_from_curr = final_ans[t]
#
#         temp = dirs_from_prev.copy()
#         for item in dirs_from_curr:
#             temp.append(item)
#
#         for item in temp:
#             if item not in ans:
#                 if temp.count(item) > 1:
#                     ans.append(item)
#
#     dirs_from_prev = final_ans[len(coords) - 1]
#     dirs_from_curr = final_ans['loop']
#
#     temp = dirs_from_prev.copy()
#     for item in dirs_from_curr:
#         temp.append(item)
#
#     for item in temp:
#         if item not in ans:
#             if temp.count(item) > 1:
#                 ans.append(item)
#
#     if PREV:
#         if PREV in ans:
#             ans.remove(PREV)
#
#     # print("ans:", ans)
#
#     return ans


def group_numbers(numbers, max_difference=7):
    groups = []
    for number in numbers:
        found_group = False
        for group in groups:
            for member in group:
                if abs(member - number) <= max_difference:
                    group.append(number)
                    found_group = True
                    break

                # remove this if-block if a number should be added to multiple groups
                if found_group:
                    break
        if not found_group:
            groups.append([number])
    return groups


# def get_dirs_from_coords(_, __, ___, X, Y, PREV):
#     # st_plot = time.time()
#     cs = plt.contour(_, __, ___, colors='black')
#     ans_f = []
#     direction_temp_count = {}
#     count_contours_dict = {}
#
#     mid_data = []
#     mid_dirs = []
#
#     # img = np.zeros((100, 100), dtype=np.uint8)
#     # img[25:75, 25:75] = 255
#     # img[40:60, 40:60] = 0
#     #
#     # plt.figure("img")
#     # plt.imshow(img)
#
#     # find the contours of the array
#     # contours, hierarchy = cv.findContours(np.array(___), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
#
#     # print the number of contours found
#     # print(f"Number of contours found: {len(contours)}")
#
#     # contours, hierarchy = cv.findContours(___, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
#     # plt.figure("contours from opencv")
#     # plt.imshow(contours)
#
#     # print("contours form opencv:", contours)
#
#     # end_plot = time.time()
#
#     # print("time for plotting:", st_plot - end_plot)
#
#     for i in range(len(cs.collections)):
#
#         flag = 0
#         index_path = 0
#
#         coords = {}
#
#         # print("something:", len(cs.collections))
#
#         count_contours = 0
#         v = None
#
#         while flag != 1:
#             try:
#                 # p = cs.collections[-3].get_paths()[index_path]
#                 p = cs.collections[i].get_paths()[index_path]
#                 # p = cs.collections[1].get_paths()[index_path]
#                 print("len(p) [X, Y]:", len(p), [X, Y])
#
#                 # plt.clf()
#                 v = p.vertices
#                 x = v[:, 0]
#                 y = v[:, 1]
#
#                 coords[index_path] = [[x[i], y[i]] for i in range(len(x))]
#
#                 # plt.figure(str([X, Y]))
#                 plt.scatter(x, y)
#
#                 count_contours += 1
#                 # print("x, y:", x, y)
#             except:
#                 flag = 1
#
#             index_path += 1
#
#         # print("coords_path:", coords, [X, Y])
#
#         # print("[X, Y] from get_dirs_from_coords:", [X, Y])
#
#         x1, y1 = [X, X], [Y - 6, Y + 6]
#         x2, y2 = [X - 6, X + 6], [Y, Y]
#         plt.plot(x1, y1, x2, y2, marker='o')
#
#         final_ans = {}
#         ans = []
#
#         angle_new_format = None
#
#         degs = []
#         degs_where = []
#         direction = []
#
#         direction_temp = None
#
#         for ii in range(len(coords)):
#             c = coords[ii]
#
#             # coords = COORDS[2]
#             # coords = COORDS[1]
#             # c = coords[0]
#
#             # print("length of COORDS:", len(COORDS))
#
#             # slopes = []
#             # degs = []
#             # degs_where = []
#             # direction = []
#
#             for i in range(1, len(c)):
#                 curr = c[i]
#                 prev = c[i - 1]
#
#                 delta_x = curr[0] - prev[0]
#                 delta_y = curr[1] - prev[1]
#                 theta_radians = np.arctan2(delta_y, delta_x)
#
#                 angle_new_format = theta_radians / np.pi
#                 degs.append(theta_radians)
#
#                 # if (curr[0] != X) and (prev[0] != X) and (curr[1] != Y) and (prev[1] != Y):
#                 if True or ((curr[0] != X) and (prev[0] != X) and (curr[1] != Y) and (prev[1] != Y)):
#                     direction_pointing = None
#
#                     if ((angle_new_format >= 0.125) and (angle_new_format <= 0.375)) or \
#                             ((angle_new_format >= -0.875) and (angle_new_format <= -0.625)):
#                         direction_pointing = 'SE'
#                     elif ((angle_new_format >= 0.375) and (angle_new_format < 0.625)) or \
#                             ((angle_new_format > -0.625) and (angle_new_format < -0.375)):
#                         direction_pointing = 'S'
#                     elif ((angle_new_format >= 0.625) and (angle_new_format <= 0.875)) or \
#                             ((angle_new_format >= -0.375) and (angle_new_format <= -0.125)):
#                         direction_pointing = 'SW'
#                     elif ((angle_new_format >= 0) and (angle_new_format < 0.125)) or \
#                             ((angle_new_format >= -1) and (angle_new_format < -0.875)) or \
#                             ((angle_new_format > 0.875) and (angle_new_format <= 1)) or \
#                             ((angle_new_format > -0.125) and (angle_new_format < 0)):
#                         # ((angle_new_format < -0.125) and (angle_new_format > 0)):
#                         direction_pointing = 'E'
#
#                     # print('direction_pointing before:', direction_pointing)
#
#                     if direction_pointing == 'SE':
#                         if (curr[1] <= Y) or (prev[1] <= Y):
#                             direction_pointing = 'NW'
#                     elif direction_pointing == 'S':
#                         if (curr[1] <= Y) or (prev[1] <= Y):
#                             direction_pointing = 'N'
#                     elif direction_pointing == 'SW':
#                         if (curr[1] <= Y) or (prev[1] <= Y):
#                             direction_pointing = 'NE'
#                     elif direction_pointing == 'E':
#                         if (curr[0] <= X) or (prev[0] <= X):
#                             direction_pointing = 'W'
#
#                     # print('direction_pointing after:', direction_pointing)
#
#                     direction.append(direction_pointing)
#                 else:
#                     direction.append(None)
#
#             print("coords:", coords, "inside get_dirs [X, Y]:", len(coords), [X, Y])
#             print("direction at:", [X, Y], direction, "inside get_dirs [X, Y]:", len(coords), [X, Y])
#
#             mid_data.append(direction)
#
#             direction_temp = [item for item in direction if item]
#
#             final_ans[ii] = list(np.unique(np.array(direction_temp)))
#
#         # print("direction_temp [X, Y]:", direction_temp, [X, Y])
#         # print("final_ans [X, Y]:", final_ans, [X, Y])
#
#         if direction_temp:
#             # # print("direction_temp at:", [X, Y], direction_temp)
#             # direction_temp_t = [direction_temp[0]]
#             #
#             # for i in range(1, len(direction_temp) - 1):
#             #     if direction_temp[i] != direction_temp[i - 1]:
#             #         direction_temp_t.append(direction_temp[i])
#             #
#             # direction_temp.append(direction_temp[-1])
#             # direction_temp = direction_temp_t
#             #
#             # # print("direction_temp trimmed at:", [X, Y], [*set(direction_temp)])
#             # print("direction_temp trimmed at:", [X, Y], direction_temp)
#             #
#             # # direction_temp = [*set(direction_temp)]
#             #
#             # dir_tbr = []
#             #
#             # for i in range(1, len(direction_temp) - 1):
#             #     item_curr = direction_temp[i]
#             #
#             #     if len(item_curr) == 2:
#             #         # if i != 0:
#             #         if (direction_temp[i - 1] in mid[item_curr]) and (direction_temp[i + 1] in mid[item_curr]):
#             #             dir_tbr.append(i)
#             #
#             # direction_temp_t = []
#             # for i in range(len(direction_temp)):
#             #     if i not in dir_tbr:
#             #         direction_temp_t.append(direction_temp[i])
#             #
#             # direction_temp = direction_temp_t
#             # direction_temp = [*set(direction_temp)]
#             #
#             # print("direction_temp trimmed and operated on at:", [X, Y], direction_temp)
#
#             # direction_temp_count = {}
#             for item in direction_temp:
#                 if item not in direction_temp_count:
#                     direction_temp_count[item] = 1
#                 else:
#                     direction_temp_count[item] += 1
#
#             # print("direction_temp_count [X, Y]:", direction_temp_count, [X, Y])
#             # direction_temp_count_array = [direction_temp_count[key] for key in direction_temp_count]
#             # direction_temp_count_mean = sum(direction_temp_count_array) / len(direction_temp_count_array)
#             # direction_temp_count_sd = np.std(direction_temp_count_array)
#             # direction_temp_count_rms = np.sqrt(
#             #     sum((item ** 2) for item in direction_temp_count_array) / len(direction_temp_count_array))
#             # direction_temp_count_cms = (sum((item ** 3) for item in direction_temp_count_array) / len(
#             #     direction_temp_count_array)) ** (1 / 3)
#             # # direction_temp_count_skew = (sum(((item - direction_temp_count_mean) ** 3) for item in direction_temp_count_array) / (len(direction_temp_count_array) * (direction_temp_count_sd ** 3)))
#             #
#             # print("direction_temp_count_mean [X, Y]:", direction_temp_count_mean, direction_temp_count_sd,
#             #       direction_temp_count_rms, direction_temp_count_cms, [X, Y])
#
#             final_ans['loop'] = final_ans[0]
#             print("coords after final_ans [X, Y]:", coords)
#
#             for t in range(1, len(coords)):
#                 dirs_from_prev = final_ans[t - 1]
#                 dirs_from_curr = final_ans[t]
#
#                 temp = dirs_from_prev.copy()
#                 for item in dirs_from_curr:
#                     temp.append(item)
#
#                 for item in temp:
#                     if item not in ans:
#                         if temp.count(item) > 1:
#                             ans.append(item)
#
#             print("ans [X, Y]:", ans, [X, Y])
#
#             if not ans:
#                 # print("v in get_dirs_from_coords():", [X, Y], v)
#
#                 for vi in range(1, len(v)):
#                     prev = v[vi - 1]
#                     curr = v[vi]
#
#                     delta_x = curr[0] - prev[0]
#                     delta_y = curr[1] - prev[1]
#                     theta_radians = np.arctan2(delta_y, delta_x)
#
#                     angle_new_format = theta_radians / np.pi
#                     degs.append(theta_radians)
#
#                     # if (curr[0] != X) and (prev[0] != X) and (curr[1] != Y) and (prev[1] != Y):
#                     if True or ((curr[0] != X) and (prev[0] != X) and (curr[1] != Y) and (prev[1] != Y)):
#                         direction_pointing = None
#
#                         if ((angle_new_format >= 0.125) and (angle_new_format <= 0.375)) or \
#                                 ((angle_new_format >= -0.875) and (angle_new_format <= -0.625)):
#                             direction_pointing = 'SE'
#                         elif ((angle_new_format >= 0.375) and (angle_new_format < 0.625)) or \
#                                 ((angle_new_format > -0.625) and (angle_new_format < -0.375)):
#                             direction_pointing = 'S'
#                         elif ((angle_new_format >= 0.625) and (angle_new_format <= 0.875)) or \
#                                 ((angle_new_format >= -0.375) and (angle_new_format <= -0.125)):
#                             direction_pointing = 'SW'
#                         elif ((angle_new_format >= 0) and (angle_new_format < 0.125)) or \
#                                 ((angle_new_format >= -1) and (angle_new_format < -0.875)) or \
#                                 ((angle_new_format > 0.875) and (angle_new_format <= 1)) or \
#                                 ((angle_new_format > -0.125) and (angle_new_format < 0)):
#                             # ((angle_new_format < -0.125) and (angle_new_format > 0)):
#                             direction_pointing = 'E'
#
#                         # print('direction_pointing before:', direction_pointing)
#
#                         if direction_pointing == 'SE':
#                             if (curr[1] <= Y) or (prev[1] <= Y):
#                                 direction_pointing = 'NW'
#                         elif direction_pointing == 'S':
#                             if (curr[1] <= Y) or (prev[1] <= Y):
#                                 direction_pointing = 'N'
#                         elif direction_pointing == 'SW':
#                             if (curr[1] <= Y) or (prev[1] <= Y):
#                                 direction_pointing = 'NE'
#                         elif direction_pointing == 'E':
#                             if (curr[0] <= X) or (prev[0] <= X):
#                                 direction_pointing = 'W'
#
#                         # print('direction_pointing after:', direction_pointing)
#
#                         direction.append(direction_pointing)
#                     else:
#                         direction.append(None)
#
#                     # print("coords [X, Y]:", coords, [X, Y])
#                     # print("direction [X, Y]:", direction, [X, Y])
#
#                 directions_extreme = {}
#                 total_extreme = 0
#
#                 for direc in direction:
#                     if direc:
#                         if direc not in directions_extreme:
#                             directions_extreme[direc] = 1
#                         else:
#                             directions_extreme[direc] += 1
#
#                     total_extreme += 1
#
#                 print("directions_extreme [X, Y]:", directions_extreme, [X, Y])
#
#                 arr_extreme = []
#                 for direc in directions_extreme:
#                     arr_extreme.append(directions_extreme[direc] / total_extreme)
#                     # arr_extreme.append(directions_extreme[direc])
#
#                 std_extreme = np.std(arr_extreme)
#
#                 print("direction if not ans:", directions_extreme, arr_extreme, std_extreme)
#
#                 ans = [key for key in directions_extreme]
#
#                 if PREV:
#                     if opp[PREV] in ans:
#                         ans.remove(opp[PREV])
#
#             dirs_from_prev = final_ans[len(coords) - 1]
#             dirs_from_curr = final_ans['loop']
#
#             temp = dirs_from_prev.copy()
#             for item in dirs_from_curr:
#                 temp.append(item)
#
#             for item in temp:
#                 if item not in ans:
#                     if temp.count(item) > 1:
#                         ans.append(item)
#
#             if PREV:
#                 if opp[PREV] in ans:
#                     ans.remove(opp[PREV])
#
#             print("ans if not ans [X, Y]:", ans, [X, Y])
#
#             # print("ans data at using contour angles:", ans, [X, Y])
#
#             # if not ans:
#             #     # print("v in get_dirs_from_coords():", [X, Y], v)
#             #
#             #     for vi in range(1, len(v)):
#             #         prev = v[vi - 1]
#             #         curr = v[vi]
#             #
#             #         delta_x = curr[0] - prev[0]
#             #         delta_y = curr[1] - prev[1]
#             #         theta_radians = np.arctan2(delta_y, delta_x)
#             #
#             #         angle_new_format = theta_radians / np.pi
#             #         degs.append(theta_radians)
#             #
#             #         # if (curr[0] != X) and (prev[0] != X) and (curr[1] != Y) and (prev[1] != Y):
#             #         if True or ((curr[0] != X) and (prev[0] != X) and (curr[1] != Y) and (prev[1] != Y)):
#             #             direction_pointing = None
#             #
#             #             if ((angle_new_format >= 0.125) and (angle_new_format <= 0.375)) or \
#             #                     ((angle_new_format >= -0.875) and (angle_new_format <= -0.625)):
#             #                 direction_pointing = 'SE'
#             #             elif ((angle_new_format >= 0.375) and (angle_new_format < 0.625)) or \
#             #                     ((angle_new_format > -0.625) and (angle_new_format < -0.375)):
#             #                 direction_pointing = 'S'
#             #             elif ((angle_new_format >= 0.625) and (angle_new_format <= 0.875)) or \
#             #                     ((angle_new_format >= -0.375) and (angle_new_format <= -0.125)):
#             #                 direction_pointing = 'SW'
#             #             elif ((angle_new_format >= 0) and (angle_new_format < 0.125)) or \
#             #                     ((angle_new_format >= -1) and (angle_new_format < -0.875)) or \
#             #                     ((angle_new_format > 0.875) and (angle_new_format <= 1)) or \
#             #                     ((angle_new_format > -0.125) and (angle_new_format < 0)):
#             # ((angle_new_format < -0.125) and (angle_new_format > 0)):
#             #                 direction_pointing = 'E'
#             #
#             #             # print('direction_pointing before:', direction_pointing)
#             #
#             #             if direction_pointing == 'SE':
#             #                 if (curr[1] <= Y) or (prev[1] <= Y):
#             #                     direction_pointing = 'NW'
#             #             elif direction_pointing == 'S':
#             #                 if (curr[1] <= Y) or (prev[1] <= Y):
#             #                     direction_pointing = 'N'
#             #             elif direction_pointing == 'SW':
#             #                 if (curr[1] <= Y) or (prev[1] <= Y):
#             #                     direction_pointing = 'NE'
#             #             elif direction_pointing == 'E':
#             #                 if (curr[0] <= X) or (prev[0] <= X):
#             #                     direction_pointing = 'W'
#             #
#             #             # print('direction_pointing after:', direction_pointing)
#             #
#             #             direction.append(direction_pointing)
#             #         else:
#             #             direction.append(None)
#             #
#             #         print("coords [X, Y]:", coords, [X, Y])
#             #         print("direction [X, Y]:", direction, [X, Y])
#             #
#             #     directions_extreme = {}
#             #     total_extreme = 0
#             #
#             #     for direc in direction:
#             #         if direc:
#             #             if direc not in directions_extreme:
#             #                 directions_extreme[direc] = 1
#             #             else:
#             #                 directions_extreme[direc] += 1
#             #
#             #         total_extreme += 1
#             #
#             #     print("directions_extreme [X, Y]:", directions_extreme, [X, Y])
#             #
#             #     arr_extreme = []
#             #     for direc in directions_extreme:
#             #         arr_extreme.append(directions_extreme[direc] / total_extreme)
#             #         # arr_extreme.append(directions_extreme[direc])
#             #
#             #     std_extreme = np.std(arr_extreme)
#             #
#             #     # dir_arr_extreme = {}
#             #     # i = 0
#             #     #
#             #     # for item in directions_extreme:
#             #     #     temp_k = arr_extreme[i]
#             #     #
#             #     #     if temp_k not in dir_arr_extreme:
#             #     #         dir_arr_extreme[temp_k] = [item]
#             #     #     else:
#             #     #         dir_arr_extreme[temp_k].append(item)
#             #     #
#             #     #     i += 1
#             #     #
#             #     # arr_extreme = [*set(arr_extreme)]
#             #
#             #     # directions_extreme = group_numbers(directions_extreme)
#             #     #
#             #     # max_index = 0
#             #     # max_value = 0
#             #     # for i in range(len(directions_extreme)):
#             #     #     temp = directions_extreme[i]
#             #     #
#             #     #     if max(temp) > max_value:
#             #     #         max_index = i
#             #     #         max_value = max(temp)
#             #     #
#             #     # directions_extreme = directions_extreme[max_index]
#             #
#             #     print("direction if not ans:", directions_extreme, arr_extreme, std_extreme)
#             #     # print("direction if not ans:", directions_extreme, arr_extreme)
#             #
#             #     # max_in_arr_extreme = max(arr_extreme)
#             #     # arr_extreme = group_numbers(arr_extreme, 0.1)
#             #     # probably_req_directions = []
#             #     #
#             #     # f = 0
#             #     # for arr_inner in arr_extreme:
#             #     #     if f == 0:
#             #     #         if max_in_arr_extreme in arr_inner:
#             #     #             arr_extreme = arr_inner
#             #     #             f = 1
#             #     #
#             #     # for item in arr_extreme:
#             #     #     probably_req_directions.append(dir_arr_extreme[item])
#             #     #
#             #     # print("direction if not ans:", directions_extreme, arr_extreme, probably_req_directions, [X, Y])
#             #
#             #     ans = [key for key in directions_extreme]
#             #
#             #     if PREV:
#             #         if opp[PREV] in ans:
#             #             ans.remove(opp[PREV])
#
#             # print("ans data at using vertex angles:", ans, [X, Y])
#
#             # temp = [item for item in direction if item]
#             #
#             # final_ans[ii] = list(np.unique(np.array(temp)))
#             #
#             # # plt.figure("v at [X, Y]" + str([X, Y]))
#             # # plt.imshow(v)
#             #
#             # final_ans['loop'] = final_ans[0]
#             #
#             # for t in range(1, len(coords)):
#             #     dirs_from_prev = final_ans[t - 1]
#             #     dirs_from_curr = final_ans[t]
#             #
#             #     temp = dirs_from_prev.copy()
#             #     for item in dirs_from_curr:
#             #         temp.append(item)
#             #
#             #     for item in temp:
#             #         if item not in ans:
#             #             if temp.count(item) > 1:
#             #                 ans.append(item)
#             #
#             # dirs_from_prev = final_ans[len(coords) - 1]
#             # dirs_from_curr = final_ans['loop']
#             #
#             # temp = dirs_from_prev.copy()
#             # for item in dirs_from_curr:
#             #     temp.append(item)
#             #
#             # for item in temp:
#             #     if item not in ans:
#             #         if temp.count(item) > 1:
#             #             ans.append(item)
#             #
#             # print("ans inside if not ans before removing PREV:", ans)
#             #
#             # if PREV:
#             #     if opp[PREV] in ans:
#             #         ans.remove(opp[PREV])
#             #
#             # print("ans inside if not ans after removing PREV:", ans)
#
#             if not ans:
#                 print("still not ans at:", [X, Y])
#
#             ans_f.append(ans)
#             # return ans, PREV, count_contours, angle_new_format
#
#         else:
#             ans_f.append(None)
#
#         if count_contours not in count_contours_dict:
#             count_contours_dict[count_contours] = 1
#
#         else:
#             count_contours_dict[count_contours] += 1
#
#     print("direction_temp_count [X, Y]:", direction_temp_count, [X, Y])
#     direction_temp_count_array = [direction_temp_count[key] for key in direction_temp_count]
#     direction_temp_count_mean = sum(direction_temp_count_array) / len(direction_temp_count_array)
#     # direction_temp_count_sd = np.std(direction_temp_count_array)
#     # direction_temp_count_rms = np.sqrt(
#     #     sum((item ** 2) for item in direction_temp_count_array) / len(direction_temp_count_array))
#     # direction_temp_count_cms = (sum((item ** 3) for item in direction_temp_count_array) / len(
#     #     direction_temp_count_array)) ** (1 / 3)
#     # # direction_temp_count_skew = (sum(((item - direction_temp_count_mean) ** 3) for item in direction_temp_count_array) / (len(direction_temp_count_array) * (direction_temp_count_sd ** 3)))
#     #
#     # print("direction_temp_count_mean [X, Y]:", direction_temp_count_mean, direction_temp_count_sd,
#     #       direction_temp_count_rms, direction_temp_count_cms, [X, Y])
#
#     direction_temp_count_t = {}
#     for key in direction_temp_count:
#         if direction_temp_count[key] > direction_temp_count_mean:
#             direction_temp_count_t[key] = direction_temp_count[key]
#
#     direction_temp_count = direction_temp_count_t
#     del direction_temp_count_t
#
#     print("direction_temp_count [X, Y] operated on:", direction_temp_count, [X, Y])
#     print("ans_f in get_dirs_from_coords() at:", [X, Y], ans_f)
#
#     print("count_contours_dict at [X, Y]:", [X, Y], count_contours_dict)
#
#     max_count_contours = 0
#     for key in count_contours_dict:
#         if count_contours_dict[key] > max_count_contours:
#             max_count_contours = count_contours_dict[key]
#
#     ans = [key for key in direction_temp_count]
#
#     # for i in range(len(mid_data)):
#     #     list_of_directions_t = [item for item in mid_data[i] if item in ans]
#     #     # list_of_directions_t = [item for item in mid_data[i] if True]
#     #
#     #     mid_data[i] = list_of_directions_t
#     #
#     #     list_of_directions_tt = [mid_data[i][0]]
#     #
#     #     for ii in range(1, len(mid_data[i])):
#     #         list_of_directions_tt.append(mid_data[i][ii]) if mid_data[i][ii] != mid_data[i][ii - 1] else None
#     #
#     #     mid_data[i] = list_of_directions_tt
#     #
#     # print("mid_data in get_dirs_from_coords() at:", [X, Y], mid_data)
#     #
#     # # ans = []
#     #
#     # for arr in mid_data:
#     #     for i in range(1, len(arr) - 1):
#     #         prev = arr[i - 1]
#     #         curr = arr[i]
#     #         next = arr[i + 1]
#     #
#     #         if (curr in fc_nxt[prev]) and (next in fc_nxt[curr]) and (prev not in mid_dirs) and (next not in mid_dirs):
#     #             mid_dirs.append(curr) if curr not in mid_dirs else None
#     #
#     #     pass
#     #
#     # for mid_dir in mid_dirs:
#     #     ans.remove(mid_dir)
#
#     # ans = find_mid(ans)
#
#     # print("finalest data from get_dirs_from_coords at:", [X, Y], ans, PREV, max_count_contours)
#
#     return ans, PREV, max_count_contours, None

def get_dirs_from_coords(_, __, ___, X, Y, PREV):
    # st_plot = time.time()
    cs = plt.contour(_, __, ___, colors='black')
    ans_f = []
    direction_temp_count = {}
    count_contours_dict = {}

    end_dir_coords = {}

    for i in range(len(cs.collections)):

        flag = 0
        index_path = 0

        coords = {}

        # print("something:", len(cs.collections))

        count_contours = 0
        v = None

        while flag != 1:
            try:
                # p = cs.collections[-3].get_paths()[index_path]
                p = cs.collections[i].get_paths()[index_path]
                # p = cs.collections[1].get_paths()[index_path]
                print("len(p) [X, Y]:", len(p), [X, Y])

                # plt.clf()
                v = p.vertices
                x = v[:, 0]
                y = v[:, 1]

                coords[index_path] = [[x[i], y[i]] for i in range(len(x))]

                # plt.figure(str([X, Y]))
                plt.scatter(x, y)

                count_contours += 1
                # print("x, y:", x, y)
            except:
                flag = 1

            index_path += 1

        # print("coords_path:", coords, [X, Y])

        # print("[X, Y] from get_dirs_from_coords:", [X, Y])

        x1, y1 = [X, X], [Y - 6, Y + 6]
        x2, y2 = [X - 6, X + 6], [Y, Y]
        plt.plot(x1, y1, x2, y2, marker='o')

        final_ans = {}
        ans = []

        angle_new_format = None

        degs = []
        degs_where = []
        direction = []

        direction_temp = None

        for ii in range(len(coords)):
            c = coords[ii]

            # coords = COORDS[2]
            # coords = COORDS[1]
            # c = coords[0]

            # print("length of COORDS:", len(COORDS))

            # slopes = []
            # degs = []
            # degs_where = []
            # direction = []

            for i in range(1, len(c)):
                curr = c[i]
                prev = c[i - 1]

                delta_x = curr[0] - prev[0]
                delta_y = curr[1] - prev[1]
                theta_radians = np.arctan2(delta_y, delta_x)

                angle_new_format = theta_radians / np.pi
                degs.append(theta_radians)

                # if (curr[0] != X) and (prev[0] != X) and (curr[1] != Y) and (prev[1] != Y):
                if True or ((curr[0] != X) and (prev[0] != X) and (curr[1] != Y) and (prev[1] != Y)):
                    direction_pointing = None

                    if ((angle_new_format >= 0.125) and (angle_new_format <= 0.375)) or \
                            ((angle_new_format >= -0.875) and (angle_new_format <= -0.625)):
                        direction_pointing = 'SE'
                    elif ((angle_new_format >= 0.375) and (angle_new_format < 0.625)) or \
                            ((angle_new_format > -0.625) and (angle_new_format < -0.375)):
                        direction_pointing = 'S'
                    elif ((angle_new_format >= 0.625) and (angle_new_format <= 0.875)) or \
                            ((angle_new_format >= -0.375) and (angle_new_format <= -0.125)):
                        direction_pointing = 'SW'
                    elif ((angle_new_format >= 0) and (angle_new_format < 0.125)) or \
                            ((angle_new_format >= -1) and (angle_new_format < -0.875)) or \
                            ((angle_new_format > 0.875) and (angle_new_format <= 1)) or \
                            ((angle_new_format > -0.125) and (angle_new_format < 0)):
                        # ((angle_new_format < -0.125) and (angle_new_format > 0)):
                        direction_pointing = 'E'

                    # print('direction_pointing before:', direction_pointing)

                    if direction_pointing == 'SE':
                        if (curr[1] <= Y) or (prev[1] <= Y):
                            direction_pointing = 'NW'
                    elif direction_pointing == 'S':
                        if (curr[1] <= Y) or (prev[1] <= Y):
                            direction_pointing = 'N'
                    elif direction_pointing == 'SW':
                        if (curr[1] <= Y) or (prev[1] <= Y):
                            direction_pointing = 'NE'
                    elif direction_pointing == 'E':
                        if (curr[0] <= X) or (prev[0] <= X):
                            direction_pointing = 'W'

                    # print('direction_pointing after:', direction_pointing)

                    if i == 1:
                        if direction_pointing not in end_dir_coords:
                            end_dir_coords[direction_pointing] = [[[round(prev[0], 0), round(prev[1], 0)], delta_y, delta_x]]

                        else:
                            end_dir_coords[direction_pointing].append([[round(prev[0], 0), round(prev[1], 0)], delta_y, delta_x])

                    if i == len(coords):
                        if direction_pointing not in end_dir_coords:
                            end_dir_coords[direction_pointing] = [[[round(curr[0], 0), round(curr[1], 0)], delta_y, delta_x]]

                        else:
                            end_dir_coords[direction_pointing].append([[round(curr[0], 0), round(curr[1], 0)], delta_y, delta_x])

                    direction.append(direction_pointing)
                else:
                    direction.append(None)

            print("coords:", coords, "inside get_dirs [X, Y]:", len(coords), [X, Y])
            print("direction at:", [X, Y], direction, "inside get_dirs [X, Y]:", len(coords), [X, Y])

            direction_temp = [item for item in direction if item]

            final_ans[ii] = list(np.unique(np.array(direction_temp)))

        # print("direction_temp [X, Y]:", direction_temp, [X, Y])
        # print("final_ans [X, Y]:", final_ans, [X, Y])

        if direction_temp:
            for item in direction_temp:
                if item not in direction_temp_count:
                    direction_temp_count[item] = 1
                else:
                    direction_temp_count[item] += 1

            final_ans['loop'] = final_ans[0]
            print("coords after final_ans [X, Y]:", coords)

            for t in range(1, len(coords)):
                dirs_from_prev = final_ans[t - 1]
                dirs_from_curr = final_ans[t]

                temp = dirs_from_prev.copy()
                for item in dirs_from_curr:
                    temp.append(item)

                for item in temp:
                    if item not in ans:
                        if temp.count(item) > 1:
                            ans.append(item)

            print("ans [X, Y]:", ans, [X, Y])

            if not ans:
                # print("v in get_dirs_from_coords():", [X, Y], v)

                for vi in range(1, len(v)):
                    prev = v[vi - 1]
                    curr = v[vi]

                    delta_x = curr[0] - prev[0]
                    delta_y = curr[1] - prev[1]
                    theta_radians = np.arctan2(delta_y, delta_x)

                    angle_new_format = theta_radians / np.pi
                    degs.append(theta_radians)

                    # if (curr[0] != X) and (prev[0] != X) and (curr[1] != Y) and (prev[1] != Y):
                    if True or ((curr[0] != X) and (prev[0] != X) and (curr[1] != Y) and (prev[1] != Y)):
                        direction_pointing = None

                        if ((angle_new_format >= 0.125) and (angle_new_format <= 0.375)) or \
                                ((angle_new_format >= -0.875) and (angle_new_format <= -0.625)):
                            direction_pointing = 'SE'
                        elif ((angle_new_format >= 0.375) and (angle_new_format < 0.625)) or \
                                ((angle_new_format > -0.625) and (angle_new_format < -0.375)):
                            direction_pointing = 'S'
                        elif ((angle_new_format >= 0.625) and (angle_new_format <= 0.875)) or \
                                ((angle_new_format >= -0.375) and (angle_new_format <= -0.125)):
                            direction_pointing = 'SW'
                        elif ((angle_new_format >= 0) and (angle_new_format < 0.125)) or \
                                ((angle_new_format >= -1) and (angle_new_format < -0.875)) or \
                                ((angle_new_format > 0.875) and (angle_new_format <= 1)) or \
                                ((angle_new_format > -0.125) and (angle_new_format < 0)):
                            # ((angle_new_format < -0.125) and (angle_new_format > 0)):
                            direction_pointing = 'E'

                        # print('direction_pointing before:', direction_pointing)

                        if direction_pointing == 'SE':
                            if (curr[1] <= Y) or (prev[1] <= Y):
                                direction_pointing = 'NW'
                        elif direction_pointing == 'S':
                            if (curr[1] <= Y) or (prev[1] <= Y):
                                direction_pointing = 'N'
                        elif direction_pointing == 'SW':
                            if (curr[1] <= Y) or (prev[1] <= Y):
                                direction_pointing = 'NE'
                        elif direction_pointing == 'E':
                            if (curr[0] <= X) or (prev[0] <= X):
                                direction_pointing = 'W'

                        # print('direction_pointing after:', direction_pointing)

                        direction.append(direction_pointing)
                    else:
                        direction.append(None)

                    # print("coords [X, Y]:", coords, [X, Y])
                    # print("direction [X, Y]:", direction, [X, Y])

                directions_extreme = {}
                total_extreme = 0

                for direc in direction:
                    if direc:
                        if direc not in directions_extreme:
                            directions_extreme[direc] = 1
                        else:
                            directions_extreme[direc] += 1

                    total_extreme += 1

                print("directions_extreme [X, Y]:", directions_extreme, [X, Y])

                arr_extreme = []
                for direc in directions_extreme:
                    arr_extreme.append(directions_extreme[direc] / total_extreme)
                    # arr_extreme.append(directions_extreme[direc])

                std_extreme = np.std(arr_extreme)

                print("direction if not ans:", directions_extreme, arr_extreme, std_extreme)

                ans = [key for key in directions_extreme]

                if PREV:
                    if opp[PREV] in ans:
                        ans.remove(opp[PREV])

            dirs_from_prev = final_ans[len(coords) - 1]
            dirs_from_curr = final_ans['loop']

            temp = dirs_from_prev.copy()
            for item in dirs_from_curr:
                temp.append(item)

            for item in temp:
                if item not in ans:
                    if temp.count(item) > 1:
                        ans.append(item)

            if PREV:
                if opp[PREV] in ans:
                    ans.remove(opp[PREV])

            print("ans if not ans [X, Y]:", ans, [X, Y])

            if not ans:
                print("still not ans at:", [X, Y])

            ans_f.append(ans)
            # return ans, PREV, count_contours, angle_new_format

        else:
            ans_f.append(None)

        if count_contours not in count_contours_dict:
            count_contours_dict[count_contours] = 1

        else:
            count_contours_dict[count_contours] += 1

    print("direction_temp_count [X, Y]:", direction_temp_count, [X, Y])
    direction_temp_count_array = [direction_temp_count[key] for key in direction_temp_count]
    direction_temp_count_mean = sum(direction_temp_count_array) / len(direction_temp_count_array)
    # direction_temp_count_sd = np.std(direction_temp_count_array)
    # direction_temp_count_rms = np.sqrt(
    #     sum((item ** 2) for item in direction_temp_count_array) / len(direction_temp_count_array))
    # direction_temp_count_cms = (sum((item ** 3) for item in direction_temp_count_array) / len(
    #     direction_temp_count_array)) ** (1 / 3)
    # # direction_temp_count_skew = (sum(((item - direction_temp_count_mean) ** 3) for item in direction_temp_count_array) / (len(direction_temp_count_array) * (direction_temp_count_sd ** 3)))
    #
    # print("direction_temp_count_mean [X, Y]:", direction_temp_count_mean, direction_temp_count_sd,
    #       direction_temp_count_rms, direction_temp_count_cms, [X, Y])

    direction_temp_count_t = {}
    for key in direction_temp_count:
        if direction_temp_count[key] > direction_temp_count_mean:
            direction_temp_count_t[key] = direction_temp_count[key]

    direction_temp_count = direction_temp_count_t
    del direction_temp_count_t

    print("direction_temp_count [X, Y] operated on:", direction_temp_count, [X, Y])
    print("ans_f in get_dirs_from_coords() at:", [X, Y], ans_f)

    print("count_contours_dict at [X, Y]:", [X, Y], count_contours_dict)

    max_count_contours = 0
    for key in count_contours_dict:
        if count_contours_dict[key] > max_count_contours:
            max_count_contours = count_contours_dict[key]

    ans = [key for key in direction_temp_count]

    end_dir_coords_t = {}

    for dir_key in end_dir_coords:
        if dir_key in ans:
            end_dir_coords_t[dir_key] = end_dir_coords[dir_key]

    end_dir_coords = end_dir_coords_t
    end_dir_coords_tt = {}
    
    for dir_key in end_dir_coords:
        temp_arr = end_dir_coords[dir_key]
        coord_max, dy_max, dx_max = [X, Y], 0, 0
        
        for coord, dy, dx in temp_arr:
            if np.linalg.norm(np.array([X, Y]) - np.array(coord)) \
                    > np.linalg.norm(np.array([X, Y]) - np.array(coord_max)):

                coord_max, dy_max, dx_max = coord, dy, dx

        end_dir_coords_tt[dir_key] = [coord_max, dy_max, dx_max]

    end_dir_coords = end_dir_coords_tt

    print("end_dir_coords in get_dirs_from_coords:", [X, Y], end_dir_coords)

    return ans, PREV, max_count_contours, None


# def recenter(curr_coords, Harris_COI_coords, angle_next, step_wanted, Harris_directions, DIRECTIONS, MODE):
#     # start_recenter = timer()
#     # ratio_match = 0
#     max_ratio_match = 0
#     max_ratio_position = []
#
#     Harris_COI_X, Harris_COI_Y = Harris_COI_coords
#     curr_X, curr_Y = curr_coords
#     angle_next = angle_next
#
#     # print("Harris directions from recenter:", Harris_directions)
#
#     step_curr = 5
#     if MODE == 1:
#         step_curr = max(abs(Harris_COI_X - curr_X), abs(Harris_COI_Y - curr_Y))
#
#     possible_curr = []
#     possible_wanted = []
#
#     for x in range(curr_X - step_wanted, curr_X + step_wanted + 1):
#         for y in range(curr_Y - step_wanted, curr_Y + step_wanted + 1):
#             possible_wanted.append([x, y])
#
#     for x in range(curr_X - step_curr, curr_X + step_curr + 1):
#         for y in range(curr_Y - step_curr, curr_Y + step_curr + 1):
#             possible_curr.append([x, y])
#
#     possible_cur_values = []
#     # possible = []
#
#     directions = DIRECTIONS
#
#     X1, Y1 = Harris_COI_X, Harris_COI_Y
#
#     if MODE == 1:
#         _, __, ___, X1, Y1 = grapher_3D(13, image_final, curr_coords[0], curr_coords[1])
#         directions, pp, len_conts, angle = get_dirs_from_coords(_, __, ___, X1, Y1, None)
#
#         beta_bar_testing, angle_testing, direction_testing, _ = dirLR(image_final, X1, Y1, 27, 100)
#
#         print("MODE == 1 in recenter:", directions, [X1, Y1])
#
#     # print("directions from recenter:", directions)
#
#     X_, Y_ = X1, Y1
#
#     flag_found = 0
#
#     req_dir_next = None
#     if ((angle_next >= 0.125) and (angle_next <= 0.375)) or \
#             ((angle_next >= -0.875) and (angle_next <= -0.625)):
#         req_dir_next = perp["SE"]
#     elif ((angle_next >= 0.375) and (angle_next < 0.625)) or \
#             ((angle_next > -0.625) and (angle_next < -0.375)):
#         req_dir_next = perp["S"]
#     elif ((angle_next >= 0.625) and (angle_next <= 0.875)) or \
#             ((angle_next >= -0.375) and (angle_next <= -0.125)):
#         req_dir_next = perp["SW"]
#     elif ((angle_next >= 0) and (angle_next < 0.125)) or \
#             ((angle_next >= -1) and (angle_next < -0.875)) or \
#             ((angle_next > 0.875) and (angle_next <= 1)) or \
#             ((angle_next < -0.125) and (angle_next > 0)):
#         req_dir_next = perp["E"]
#
#     for position in possible_curr:
#         if flag_found == 0:
#             # for position in possible_wanted:
#             delta_x = curr_coords[0] - position[0]
#             delta_y = curr_coords[1] - position[1]
#             theta_radians = np.arctan2(delta_y, delta_x)
#
#             angle_new_format = theta_radians / np.pi
#
#             # perp1 = angle_next - 0.5
#             # perp2 = angle_next + 0.5
#             # # perp2 = 0.5 - angle_next
#             #
#             # # TODO: Uncomment to filter the angle (might need re-evaluation)
#             # if perp1 <= -1:
#             #     perp1 = 1 - (((-1) * perp1) % 1)
#             # elif perp1 >= 1:
#             #     perp1 = (perp1 % 1) - 1
#             #
#             # if perp2 <= -1:
#             #     perp2 = 1 - (((-1) * perp2) % 1)
#             # elif perp2 >= 1:
#             #     perp2 = (perp2 % 1) - 1
#
#             # possible_cur_values.append(abs(angle_new_format - angle_next))
#
#             angle_new_format_direction = None
#
#             if ((angle_new_format >= 0.125) and (angle_new_format <= 0.375)) or \
#                     ((angle_new_format >= -0.875) and (angle_new_format <= -0.625)):
#                 angle_new_format_direction = "SE"
#             elif ((angle_new_format >= 0.375) and (angle_new_format < 0.625)) or \
#                     ((angle_new_format > -0.625) and (angle_new_format < -0.375)):
#                 angle_new_format_direction = "S"
#             elif ((angle_new_format >= 0.625) and (angle_new_format <= 0.875)) or \
#                     ((angle_new_format >= -0.375) and (angle_new_format <= -0.125)):
#                 angle_new_format_direction = "SW"
#             elif ((angle_new_format >= 0) and (angle_new_format < 0.125)) or \
#                     ((angle_new_format >= -1) and (angle_new_format < -0.875)) or \
#                     ((angle_new_format > 0.875) and (angle_new_format <= 1)) or \
#                     ((angle_new_format > -0.125) and (angle_new_format < 0)):
# ((angle_new_format < -0.125) and (angle_new_format > 0)):
#                 angle_new_format_direction = "E"
#
#             # print("angle_new_format, angle_next:", angle_new_format, angle_next)
#
#             # if (angle_new_format == perp1) or (angle_new_format == perp2):
#             if angle_new_format_direction == req_dir_next:
#                 # dirrs, prob_arr, n = kernel_determiner(image_final, position[0], position[1], 0.46, 0.3581)
#                 dirrs, prob_arr, __ = kernel_determiner(image_final, position[0], position[1], 1.1, 0.901)
#                 # print("dirrs dirrs:", dirrs, position)
#                 # print("dirrs __:", __, position)
#
#                 intersection = np.intersect1d(__, directions)
#                 ratio = len(intersection) / len(directions)
#
#                 if ratio > max_ratio_match:
#                     max_ratio_match = ratio
#                     max_ratio_position = position
#
#                 if __ == directions:
#                     # if (Harris_directions[0] in __) and (opp[prev] in __):
#                     X_, Y_ = position[0], position[1]
#
#                     print("dirrs dirrs:", dirrs, position)
#                     print("dirrs __:", __, position, directions)
#
#                     print("---------------------------------------------------------")
#
#                     flag_found = 1
#
#                 # if Harris_directions[0] in __:
#                 #     possible.append(position)
#
#     print("data from recenter - ratio, coords, directions, :", max_ratio_match, curr_coords, directions)
#
#     if flag_found == 0:
#         if max_ratio_match != 0:
#             X_, Y_ = max_ratio_position[0], max_ratio_position[1]
#         else:
#             Y_, X_ = kernel_operator(Harris_directions[0], Harris_COI_Y, Harris_COI_X, getstep(Harris_directions[0],
#                                                                                                Harris_COI_Y,
#                                                                                                Harris_COI_X,
#                                                                                                coords_for_COI))
#
#     # minimum = min(possible_cur_values)
#     # possible = [possible_curr[i] for i in range(len(possible_curr)) if possible_cur_values[i] == minimum]
#
#     # print("possible:", possible, Harris_directions, possible_curr)
#     # print("---------------------------------------------------------")
#
#     # print("from recenter:", [X_, Y_])
#
#     # X_new, Y_new = curr_X, curr_Y
#     #
#     # return X_new, Y_new
#
#     # end_recenter = timer()
#     # print("time for recenter:", end_recenter - start_recenter)
#     return X_, Y_
#     # return X_, Y_, max_ratio_match

def find_max_perp(point, beta_bar, img_f, limit):
    # _, __, ___, X1, Y1 = grapher_3D(13, img_f, point[0], point[1])
    # return [X1, Y1]

    start_fmp = timer()

    possible = []

    xc, yc = point

    l_xc, u_xc = xc, xc
    l_yc, u_yc = yc, yc
    l_yc_d, u_yc_d = yc, yc

    range_of_points = []

    while img_f[yc][l_xc - 1] < limit:
        l_xc -= 1

    while img_f[yc][u_xc + 1] < limit:
        u_xc += 1

    while img_f[l_yc - 1][l_xc] < limit:
        l_yc -= 1

    while img_f[u_yc + 1][l_xc] < limit:
        u_yc += 1

    while img_f[l_yc_d - 1][u_xc] < limit:
        l_yc_d -= 1

    while img_f[u_yc_d + 1][u_xc] < limit:
        u_yc_d += 1

    for y in range(l_yc, u_yc + 1):
        for x in range(l_xc, u_xc + 1):
            range_of_points.append([x, y])

    for y in range(l_yc_d, u_yc_d + 1):
        for x in range(l_xc, u_xc + 1):
            range_of_points.append([x, y])

    b = np.abs(u_yc - l_yc)
    a = np.abs(u_xc - l_xc)

    thickness = np.ceil(b * np.sin(np.arctan(a / b)))
    print("data from find_max_perp at pre:", point, thickness, a, b)
    thickness /= 2

    # print("data from find_max_perp at range_possible:", range_of_points)

    if beta_bar == 0:
        beta_perp = "inf"

    else:
        beta_perp = (-1) / beta_bar

    for x_perp, y_final in range_of_points:
        y_perp = ((beta_perp * (x_perp - point[0])) + point[1]) if beta_perp != "inf" else y_final

        if ((y_final == np.floor(y_perp)) or (y_final == np.ceil(y_perp))) and img_f[y_final][x_perp] < limit:
            possible.append([x_perp, y_final])

    print("all data from find_max_perp:", u_xc, l_xc, u_yc, l_yc, "possible:", possible)

    median_from_possible = None

    for p in possible:
        if np.floor(np.linalg.norm(np.array(point) - np.array(p))) <= thickness:
            if (median_from_possible is not None) and (
                    np.linalg.norm(np.array(point) - np.array(p)) > np.linalg.norm(
                    np.array(point) - np.array(median_from_possible))):
                median_from_possible = p

            elif median_from_possible is None:
                median_from_possible = p

    print("data after first median_from_possible operation:", median_from_possible)

    if median_from_possible is None:
        for p in possible:
            if np.floor(np.linalg.norm(np.array(point) - np.array(p))) >= thickness:
                if (median_from_possible is not None) and (
                        np.linalg.norm(np.array(point) - np.array(p)) < np.linalg.norm(
                        np.array(point) - np.array(median_from_possible))):
                    median_from_possible = p

                elif median_from_possible is None:
                    median_from_possible = p

        print("data after first median_from_possible operation if it was None:", median_from_possible)

    print("data from find_max_perp at:", point, "thickness:", thickness, "final point:", median_from_possible, "beta_perp:", beta_perp)

    # ------------------------------------------------------------------------------------------------------------------

    end_fmp = timer()
    print("time taken for fmp:", end_fmp - start_fmp)

    return median_from_possible


def dirLR(IMG_F, CX, CY, KER_SIZE, LIMIT, PREV=None):
    start_dirLR = timer()

    step = (KER_SIZE - 1) // 2
    values = {}
    values_coord = []

    flag_COI = False

    x = []
    y = []
    vals = []

    # Q1 = []
    # Q2 = []
    # Q3 = []
    # Q4 = []

    flag_curr_Harris = False

    if type(maintainer_matrix[CY][CX]) == str:
        if maintainer_matrix[CY][CX][0] == "h":
            flag_curr_Harris = True

            index_mm = int(maintainer_matrix[CY][CX][1])

            for xH, yH in coords_for_groups[index_mm]:
                maintainer_matrix[yH][xH] = index_mm

            maintainer_matrix[coords_for_COI[index_mm][1]][coords_for_COI[index_mm][0]] = index_mm

    for Y in range(CY - step, CY + step + 1):
        for X in range(CX - step, CX + step + 1):
            value = IMG_F[Y][X]
            values[(X, Y)] = value

            if flag_curr_Harris == False:
                val_mm = maintainer_matrix[Y][X]
                if type(val_mm) == str:
                    if val_mm[0] == "h":
                        index_mm = int(val_mm[1])

                        print("Harris at:", [X, Y], coords_for_COI[index_mm], index_mm, type(index_mm), type(val_mm))

                        for xH, yH in coords_for_groups[index_mm]:
                            maintainer_matrix[yH][xH] = index_mm

                        maintainer_matrix[coords_for_COI[index_mm][1]][coords_for_COI[index_mm][0]] = index_mm

                        direction_pointing = None
                        curr = coords_for_COI[index_mm]

                        theta_radians = np.arctan2(curr[1] - CY, curr[0] - CX)

                        angle_new_format = theta_radians / np.pi

                        if ((angle_new_format >= 0.125) and (angle_new_format <= 0.375)) or \
                                ((angle_new_format >= -0.875) and (angle_new_format <= -0.625)):
                            direction_pointing = 'SE'
                        elif ((angle_new_format >= 0.375) and (angle_new_format < 0.625)) or \
                                ((angle_new_format > -0.625) and (angle_new_format < -0.375)):
                            direction_pointing = 'S'
                        elif ((angle_new_format >= 0.625) and (angle_new_format <= 0.875)) or \
                                ((angle_new_format >= -0.375) and (angle_new_format <= -0.125)):
                            direction_pointing = 'SW'
                        elif ((angle_new_format >= 0) and (angle_new_format < 0.125)) or \
                                ((angle_new_format >= -1) and (angle_new_format < -0.875)) or \
                                ((angle_new_format > 0.875) and (angle_new_format <= 1)) or \
                                ((angle_new_format > -0.125) and (angle_new_format < 0)):
                            # ((angle_new_format < -0.125) and (angle_new_format > 0)):
                            direction_pointing = 'E'

                        # print('direction_pointing before:', direction_pointing)

                        if direction_pointing == 'SE':
                            if curr[1] <= CY:
                                direction_pointing = 'NW'
                        elif direction_pointing == 'S':
                            if curr[1] <= CY:
                                direction_pointing = 'N'
                        elif direction_pointing == 'SW':
                            if curr[1] <= CY:
                                direction_pointing = 'NE'
                        elif direction_pointing == 'E':
                            if curr[0] <= CX:
                                direction_pointing = 'W'

                        return None, None, direction_pointing, coords_for_COI[index_mm]

            # for key in coords_for_groups:
            #     for XK, YK in coords_for_groups[key]:
            #         if flag_COI == False:
            #             if [XK, YK] == [X, Y]:
            #                 flag_COI = key
            #
            # if (flag_COI != False) and ([CX, CY] != coords_for_COI[flag_COI]):
            #     return None, None, None, coords_for_COI[flag_COI]

            # _, __, ___, X1, Y1 = grapher_3D(13, IMG_F, X, Y)
            #
            # if [X1, Y1] != [X, Y]:
            #     return None, None, None, None

            # if value < LIMIT:
            #     x.append(X)
            #     y.append(Y)
            #     vals.append((255 - values[(X, Y)]) / 255)
            #
            #     values_coord.append([X, Y])

    for key in values:
        if values[key] < LIMIT:
            x.append(key[0])
            y.append(key[1])
            vals.append((255 - values[key]) / 255)

            values_coord.append([key[0], key[1]])

            # xx = key[0]
            # yy = key[1]
            #
            # if (xx > CX) and (yy < CY):
            #     Q1.append([xx, yy])
            # elif (xx > CX) and (yy > CY):
            #     Q4.append([xx, yy])
            # elif (xx < CX) and (yy < CY):
            #     Q2.append([xx, yy])
            # elif (xx < CX) and (yy > CY):
            #     Q3.append([xx, yy])
            # elif (xx == CX):
            #     if (yy < CY):
            #         Q1.append([xx, yy])
            #         Q2.append([xx, yy])
            #     elif (yy > CY):
            #         Q3.append([xx, yy])
            #         Q4.append([xx, yy])
            # elif (yy == CY):
            #     if (xx < CX):
            #         Q2.append([xx, yy])
            #         Q3.append([xx, yy])
            #     elif (xx > CX):
            #         Q1.append([xx, yy])
            #         Q4.append([xx, yy])
            # elif (xx == CX) and (yy == CY):
            #     Q1.append([xx, yy])
            #     Q2.append([xx, yy])
            #     Q3.append([xx, yy])
            #     Q4.append([xx, yy])

    wmean_x, wmean_y = 0, 0

    for i in range(len(vals)):
        wmean_x += (vals[i] * x[i])
        wmean_y += (vals[i] * y[i])

    xmean = wmean_x / sum(vals)
    ymean = wmean_y / sum(vals)

    beta_bar_num, beta_bar_denom = 0, 0

    for i in range(len(vals)):
        beta_bar_num += ((x[i] - xmean) * (y[i] - ymean))

    for i in range(len(vals)):
        beta_bar_denom += ((x[i] - xmean) ** 2)

    beta_bar = beta_bar_num / beta_bar_denom
    angle = np.arctan2(beta_bar_num, beta_bar_denom) / np.pi

    points_possible = []
    # points_possible_Q1 = []
    # points_possible_Q2 = []
    # points_possible_Q3 = []
    # points_possible_Q4 = []

    # print("values_coord:", values_coord)

    # for y in range(CY - step, CY + step + 1):
    #     for x in range(CX - step, CX + step + 1):

    print("values_coord in dirLR:", [CX, CY], values_coord)
    print("vals in dirLR:", [CX, CY], vals)

    # print("Q1 in dirLR:", [CX, CY], Q1)
    # print("Q2 in dirLR:", [CX, CY], Q2)
    # print("Q3 in dirLR:", [CX, CY], Q3)
    # print("Q4 in dirLR:", [CX, CY], Q4)

    if beta_bar == 0:
        beta_bar = 10e-10

    if True:
        for x, y in values_coord:
            try:
                # print("error in divide by 0:", beta_bar, beta_bar_num, beta_bar_denom)

                lower_x = np.floor(((y - CY) / beta_bar) + CX)
                # print("inside np.floor:", y - CY, beta_bar)
                # upper_x = np.ceil(((y - CY) / beta_bar) + CX)
                upper_x = lower_x + 1
                # value = ((y - CY) / beta_bar) + CX
                #
                # if (value - lower_x) >= (upper_x - value):
                #     x_final = upper_x
                # else:
                #     x_final = lower_x

                # if (x == lower_x) or (x == upper_x):
                #     points_possible.append([x, y])

                # if x == x_final:
                #     points_possible.append([x, y])

                if (x == upper_x) or (x == lower_x):
                    points_possible.append([x, y])

                lower_y = np.floor(((x - CX) * beta_bar) + CY)
                upper_y = lower_y + 1

                if (y == upper_y) or (y == lower_y):
                    points_possible.append([x, y])

            except:
                print("error in divide by 0:", beta_bar, beta_bar_num, beta_bar_denom)

    # if True:
    #     for x, y in Q1:
    #         try:
    #             # print("error in divide by 0:", beta_bar, beta_bar_num, beta_bar_denom)
    #
    #             lower_x = np.floor(((y - CY) / beta_bar) + CX)
    #             upper_x = lower_x + 1
    #
    #             if (x == upper_x) or (x == lower_x):
    #                 points_possible.append([x, y])
    #
    #             lower_y = np.floor(((x - CX) * beta_bar) + CY)
    #             upper_y = lower_y + 1
    #
    #             if (y == upper_y) or (y == lower_y):
    #                 points_possible_Q1.append([x, y])
    #
    #         except:
    #             print("error in divide by 0 in Q1:", beta_bar, beta_bar_num, beta_bar_denom)
    #
    #     for x, y in Q2:
    #         try:
    #             # print("error in divide by 0:", beta_bar, beta_bar_num, beta_bar_denom)
    #
    #             lower_x = np.floor(((y - CY) / beta_bar) + CX)
    #             upper_x = lower_x + 1
    #
    #             if (x == upper_x) or (x == lower_x):
    #                 points_possible.append([x, y])
    #
    #             lower_y = np.floor(((x - CX) * beta_bar) + CY)
    #             upper_y = lower_y + 1
    #
    #             if (y == upper_y) or (y == lower_y):
    #                 points_possible_Q2.append([x, y])
    #
    #         except:
    #             print("error in divide by 0 in Q1:", beta_bar, beta_bar_num, beta_bar_denom)
    #
    #     for x, y in Q3:
    #         try:
    #             # print("error in divide by 0:", beta_bar, beta_bar_num, beta_bar_denom)
    #
    #             lower_x = np.floor(((y - CY) / beta_bar) + CX)
    #             upper_x = lower_x + 1
    #
    #             if (x == upper_x) or (x == lower_x):
    #                 points_possible.append([x, y])
    #
    #             lower_y = np.floor(((x - CX) * beta_bar) + CY)
    #             upper_y = lower_y + 1
    #
    #             if (y == upper_y) or (y == lower_y):
    #                 points_possible_Q3.append([x, y])
    #
    #         except:
    #             print("error in divide by 0 in Q1:", beta_bar, beta_bar_num, beta_bar_denom)
    #
    #     for x, y in Q4:
    #         try:
    #             # print("error in divide by 0:", beta_bar, beta_bar_num, beta_bar_denom)
    #
    #             lower_x = np.floor(((y - CY) / beta_bar) + CX)
    #             upper_x = lower_x + 1
    #
    #             if (x == upper_x) or (x == lower_x):
    #                 points_possible.append([x, y])
    #
    #             lower_y = np.floor(((x - CX) * beta_bar) + CY)
    #             upper_y = lower_y + 1
    #
    #             if (y == upper_y) or (y == lower_y):
    #                 points_possible_Q4.append([x, y])
    #
    #         except:
    #             print("error in divide by 0 in Q1:", beta_bar, beta_bar_num, beta_bar_denom)

    # direction_list = []
    degs = []

    print("points_possible in dirLR:", [CX, CY], points_possible)
    # print("points_possible in dirLR Q1:", [CX, CY], points_possible_Q1)
    # print("points_possible in dirLR Q2:", [CX, CY], points_possible_Q2)
    # print("points_possible in dirLR Q3:", [CX, CY], points_possible_Q3)
    # print("points_possible in dirLR Q4:", [CX, CY], points_possible_Q4)

    # for i in range(1, len(points_possible)):
    #     curr = points_possible[i]
    #     prev = points_possible[i - 1]
    #
    #     delta_x = curr[0] - prev[0]
    #     delta_y = curr[1] - prev[1]
    #     theta_radians = np.arctan2(delta_y, delta_x)
    #
    #     angle_new_format = theta_radians / np.pi
    #     degs.append(theta_radians)
    #
    #     # if (curr[0] != X) and (prev[0] != X) and (curr[1] != Y) and (prev[1] != Y):
    #     if True or ((curr[0] != XO) and (prev[0] != XO) and (curr[1] != YO) and (prev[1] != YO)):
    #         direction_pointing = None
    #
    #         if ((angle_new_format >= 0.125) and (angle_new_format <= 0.375)) or \
    #                 ((angle_new_format >= -0.875) and (angle_new_format <= -0.625)):
    #             direction_pointing = 'SE'
    #         elif ((angle_new_format >= 0.375) and (angle_new_format < 0.625)) or \
    #                 ((angle_new_format > -0.625) and (angle_new_format < -0.375)):
    #             direction_pointing = 'S'
    #         elif ((angle_new_format >= 0.625) and (angle_new_format <= 0.875)) or \
    #                 ((angle_new_format >= -0.375) and (angle_new_format <= -0.125)):
    #             direction_pointing = 'SW'
    #         elif ((angle_new_format >= 0) and (angle_new_format < 0.125)) or \
    #                 ((angle_new_format >= -1) and (angle_new_format < -0.875)) or \
    #                 ((angle_new_format > 0.875) and (angle_new_format <= 1)) or \
    #                 ((angle_new_format > -0.125) and (angle_new_format < 0)):
    #             # ((angle_new_format < -0.125) and (angle_new_format > 0)):
    #             direction_pointing = 'E'
    #
    #         # print('direction_pointing before:', direction_pointing)
    #
    #         if direction_pointing == 'SE':
    #             if (curr[1] <= CY) or (prev[1] <= CY):
    #                 direction_pointing = 'NW'
    #         elif direction_pointing == 'S':
    #             if (curr[1] <= CY) or (prev[1] <= CY):
    #                 direction_pointing = 'N'
    #         elif direction_pointing == 'SW':
    #             if (curr[1] <= CY) or (prev[1] <= CY):
    #                 direction_pointing = 'NE'
    #         elif direction_pointing == 'E':
    #             if (curr[0] <= CX) or (prev[0] <= CX):
    #                 direction_pointing = 'W'
    #
    #         # print('direction_pointing after:', direction_pointing)
    #
    #         direction_list.append(direction_pointing)
    #     else:
    #         direction_list.append(None)
    #
    # direction_list = [*set(direction_list)]
    direction_list_temp = []
    direction_list_coord = []
    # direction_list_coord = defaultdict(list)

    # print("direction_list in dirLR:", [CX, CY], direction_list)

    direction_general = None

    if ((angle >= 0.125) and (angle <= 0.375)) or \
            ((angle >= -0.875) and (angle <= -0.625)):
        direction_general = 'SE'
    elif ((angle >= 0.375) and (angle < 0.625)) or \
            ((angle > -0.625) and (angle < -0.375)):
        direction_general = 'S'
    elif ((angle >= 0.625) and (angle <= 0.875)) or \
            ((angle >= -0.375) and (angle <= -0.125)):
        direction_general = 'SW'
    elif ((angle >= 0) and (angle < 0.125)) or \
            ((angle >= -1) and (angle < -0.875)) or \
            ((angle > 0.875) and (angle <= 1)) or \
            ((angle > -0.125) and (angle < 0)):
        # ((angle < -0.125) and (angle > 0)):

        direction_general = 'E'

    # if len(direction_list) > 0:
    if True:
        # if PREV:
        #     PREV = opp[PREV]

        print("inside length > 0 with points_possible:", points_possible, direction_general, angle, beta_bar)

        # extremes = [points_possible[0], points_possible[-1]]
        # points_possible = []

        # for i in range(2):
        for curr in points_possible:
            # curr = extremes[i]
            prev = [CX, CY]

            delta_x = curr[0] - prev[0]
            delta_y = curr[1] - prev[1]
            theta_radians = np.arctan2(delta_y, delta_x)

            angle_new_format = theta_radians / np.pi

            # if (curr[0] != X) and (prev[0] != X) and (curr[1] != Y) and (prev[1] != Y):
            if True or ((curr[0] != XO) and (prev[0] != XO) and (curr[1] != YO) and (prev[1] != YO)):
                direction_pointing = None

                if ((angle_new_format >= 0.125) and (angle_new_format <= 0.375)) or \
                        ((angle_new_format >= -0.875) and (angle_new_format <= -0.625)):
                    direction_pointing = 'SE'
                elif ((angle_new_format >= 0.375) and (angle_new_format < 0.625)) or \
                        ((angle_new_format > -0.625) and (angle_new_format < -0.375)):
                    direction_pointing = 'S'
                elif ((angle_new_format >= 0.625) and (angle_new_format <= 0.875)) or \
                        ((angle_new_format >= -0.375) and (angle_new_format <= -0.125)):
                    direction_pointing = 'SW'
                elif ((angle_new_format >= 0) and (angle_new_format < 0.125)) or \
                        ((angle_new_format >= -1) and (angle_new_format < -0.875)) or \
                        ((angle_new_format > 0.875) and (angle_new_format <= 1)) or \
                        ((angle_new_format > -0.125) and (angle_new_format < 0)):
                    # ((angle_new_format < -0.125) and (angle_new_format > 0)):
                    direction_pointing = 'E'

                # print('direction_pointing before:', direction_pointing)

                if direction_pointing == direction_general:
                    # if True:

                    if direction_pointing == 'SE':
                        if (curr[1] <= CY):
                            direction_pointing = 'NW'
                    elif direction_pointing == 'S':
                        if (curr[1] <= CY):
                            direction_pointing = 'N'
                    elif direction_pointing == 'SW':
                        if (curr[1] <= CY):
                            direction_pointing = 'NE'
                    elif direction_pointing == 'E':
                        if (curr[0] <= CX):
                            direction_pointing = 'W'

                    # if PREV and (direction_pointing == PREV):
                    #     continue

                    # if direction_pointing in direction_list:
                    if True:
                        # points_possible.append(curr)

                        if direction_pointing not in direction_list_temp:
                            direction_list_temp.append(direction_pointing)
                            direction_list_coord.append(curr)
                            # direction_list_coord[direction_pointing].append(curr)

                        else:
                            try:
                                index_direction_list_temp = direction_list_temp.index(direction_pointing)

                                curr_temp = direction_list_coord[index_direction_list_temp]
                                # curr_temp = direction_list_coord[direction_pointing]
                                dist_temp = np.sqrt(((curr_temp[0] - CX) ** 2) + ((curr_temp[1] - CY) ** 2))
                                dist_curr = np.sqrt(((curr[0] - CX) ** 2) + ((curr[1] - CY) ** 2))

                                if dist_temp < dist_curr:
                                    direction_list_coord[index_direction_list_temp] = curr
                                    # direction_list_coord[direction_pointing] = curr

                            except:
                                print("error in direction_list_coord:", direction_list_coord)

                    # print("direction_list_coord in dirLR:", direction_pointing, curr, [CX, CY], direction_list_coord)

        points_possible = direction_list_coord

    # print("points_possible, direction_list from dirLR at:", [CX, CY], direction_list, direction_list_temp, points_possible, flag_COI, direction_general, angle)
    print("points_possible, direction_list from dirLR at:", [CX, CY], direction_list_temp, points_possible, flag_COI,
          direction_general, angle)

    # ----------------------------------------------

    final_point = []
    final_direction = []

    # if len(points_possible) == 1:
    #     final_point = points_possible[0]
    #     final_direction = direction_list[0]
    #
    # else:
    if True:
        print("points_possible from dirLR:", [CX, CY], points_possible, direction_list_temp, PREV)

        for i in range(len(direction_list_temp)):
            direction = direction_list_temp[i]
            if PREV and (direction != opp[PREV]):
                final_direction.append(direction)

                final_point.append(find_max_perp(points_possible[i], beta_bar, IMG_F, LIMIT))
                # final_point.append(points_possible[i])

            elif not PREV:
                final_direction = direction_list_temp
                # final_point = points_possible
                final_point = []

                for point in points_possible:
                    final_point.append(find_max_perp(point, beta_bar, IMG_F, LIMIT))

    print("final data from dirLR at:", [CX, CY], final_direction, final_point, beta_bar)

    # ----------------------------------------------

    # # ßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßß
    #
    # for dp in values_coord:
    #     if dp not in final_point:
    #         maintainer_matrix[dp[1]][dp[0]] = "d"
    #
    # final_point_t = []
    #
    # for fp_x, fp_y in final_point:
    #     if maintainer_matrix[fp_y][fp_x] != "d":
    #         final_point_t.append([fp_x, fp_y])
    #
    # final_point = final_point_t
    #
    # # ßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßß

    end_dirLR = timer()
    print("time taken for dirLR:", end_dirLR - start_dirLR)

    # return beta_bar, angle, direction, None
    return beta_bar, angle, final_direction, final_point

def dirLR_β(IMG_F, CX, CY, KER_SIZE, LIMIT, PREV, PREV_COORDS=None):
    start_dirLR = timer()

    step = (KER_SIZE - 1) // 2
    values = {}
    values_coord = []

    flag_COI = False

    x = []
    y = []
    vals = []

    Q1 = []
    Q2 = []
    Q3 = []
    Q4 = []

    flag_curr_Harris = False

    if type(maintainer_matrix[CY][CX]) == str:
        if maintainer_matrix[CY][CX][0] == "h":
            flag_curr_Harris = True

            index_mm = int(maintainer_matrix[CY][CX][1])

            for xH, yH in coords_for_groups[index_mm]:
                maintainer_matrix[yH][xH] = index_mm

            maintainer_matrix[coords_for_COI[index_mm][1]][coords_for_COI[index_mm][0]] = index_mm

    for Y in range(CY - step, CY + step + 1):
        for X in range(CX - step, CX + step + 1):
            value = IMG_F[Y][X]
            values[(X, Y)] = value

    for key in values:
        if values[key] < LIMIT:
        # if True or (values[key] < LIMIT):
            x.append(key[0])
            y.append(key[1])
            vals.append((255 - values[key]) / 255)

            values_coord.append([key[0], key[1]])

            xx = key[0]
            yy = key[1]

            if (xx > CX) and (yy < CY):
                Q1.append([xx, yy, (255 - values[key]) / 255])
            elif (xx > CX) and (yy > CY):
                Q4.append([xx, yy, (255 - values[key]) / 255])
            elif (xx < CX) and (yy < CY):
                Q2.append([xx, yy, (255 - values[key]) / 255])
            elif (xx < CX) and (yy > CY):
                Q3.append([xx, yy, (255 - values[key]) / 255])
            elif (xx == CX):
                if (yy < CY):
                    Q1.append([xx, yy, (255 - values[key]) / 255])
                    Q2.append([xx, yy, (255 - values[key]) / 255])
                elif (yy > CY):
                    Q3.append([xx, yy, (255 - values[key]) / 255])
                    Q4.append([xx, yy, (255 - values[key]) / 255])
            elif (yy == CY):
                if (xx < CX):
                    Q2.append([xx, yy, (255 - values[key]) / 255])
                    Q3.append([xx, yy, (255 - values[key]) / 255])
                elif (xx > CX):
                    Q1.append([xx, yy, (255 - values[key]) / 255])
                    Q4.append([xx, yy, (255 - values[key]) / 255])
            elif (xx == CX) and (yy == CY):
                Q1.append([xx, yy, (255 - values[key]) / 255])
                Q2.append([xx, yy, (255 - values[key]) / 255])
                Q3.append([xx, yy, (255 - values[key]) / 255])
                Q4.append([xx, yy, (255 - values[key]) / 255])

    # wmean_x, wmean_y = 0, 0
    wmean_x_fQS, wmean_y_fQS = 0, 0

    final_QS = []

    if PREV == opp["NE"]:
        final_QS = Q2 + Q3 + Q4

    elif PREV == opp["E"]:
        final_QS = Q2 + Q3

    elif PREV == opp["SE"]:
        final_QS = Q1 + Q2 + Q3

    elif PREV == opp["S"]:
        final_QS = Q1 + Q2

    elif PREV == opp["SW"]:
        final_QS = Q1 + Q2 + Q4

    elif PREV == opp["W"]:
        final_QS = Q1 + Q4

    elif PREV == opp["NW"]:
        final_QS = Q1 + Q3 + Q4

    elif PREV == opp["N"]:
        final_QS = Q3 + Q4

    elif not PREV:
        final_QS = Q1 + Q2 + Q3 + Q4

    # for x, y, _ in final_QS:
    #     if maintainer_matrix[y][x] == "v":
    #         maintainer_matrix[CY][CX] = "v"
    #
    #         # for xx, yy, __ in final_QS:
    #         #     maintainer_matrix[yy][xx] = "v"
    #
    #         return None, None, None, None

    for X, Y, VAL in final_QS:
        if flag_curr_Harris == False:
            val_mm = maintainer_matrix[Y][X]
            if type(val_mm) == str:
                if val_mm[0] == "h":
                    index_mm = int(val_mm[1])

                    print("Harris at:", [X, Y], coords_for_COI[index_mm], index_mm, type(index_mm), type(val_mm))

                    for xH, yH in coords_for_groups[index_mm]:
                        maintainer_matrix[yH][xH] = index_mm

                    maintainer_matrix[coords_for_COI[index_mm][1]][coords_for_COI[index_mm][0]] = index_mm

                    direction_pointing = None
                    curr = coords_for_COI[index_mm]

                    theta_radians = np.arctan2(curr[1] - CY, curr[0] - CX)

                    angle_new_format = theta_radians / np.pi

                    if ((angle_new_format >= 0.125) and (angle_new_format <= 0.375)) or \
                            ((angle_new_format >= -0.875) and (angle_new_format <= -0.625)):
                        direction_pointing = 'SE'
                    elif ((angle_new_format >= 0.375) and (angle_new_format < 0.625)) or \
                            ((angle_new_format > -0.625) and (angle_new_format < -0.375)):
                        direction_pointing = 'S'
                    elif ((angle_new_format >= 0.625) and (angle_new_format <= 0.875)) or \
                            ((angle_new_format >= -0.375) and (angle_new_format <= -0.125)):
                        direction_pointing = 'SW'
                    elif ((angle_new_format >= 0) and (angle_new_format < 0.125)) or \
                            ((angle_new_format >= -1) and (angle_new_format < -0.875)) or \
                            ((angle_new_format > 0.875) and (angle_new_format <= 1)) or \
                            ((angle_new_format > -0.125) and (angle_new_format < 0)):
                        # ((angle_new_format < -0.125) and (angle_new_format > 0)):
                        direction_pointing = 'E'

                    # print('direction_pointing before:', direction_pointing)

                    if direction_pointing == 'SE':
                        if curr[1] <= CY:
                            direction_pointing = 'NW'
                    elif direction_pointing == 'S':
                        if curr[1] <= CY:
                            direction_pointing = 'N'
                    elif direction_pointing == 'SW':
                        if curr[1] <= CY:
                            direction_pointing = 'NE'
                    elif direction_pointing == 'E':
                        if curr[0] <= CX:
                            direction_pointing = 'W'

                    return None, 'angle', direction_pointing, coords_for_COI[index_mm]

    # for i in range(len(vals)):
    #     wmean_x += (vals[i] * x[i])
    #     wmean_y += (vals[i] * y[i])

    for xi, yi, val in final_QS:
        wmean_x_fQS += (val * xi)
        wmean_y_fQS += (val * yi)

    print("final_QS in dirLR:", final_QS, PREV)

    # xmean = wmean_x / sum(vals)
    # ymean = wmean_y / sum(vals)

    xmean_fQS = wmean_x_fQS / sum(item[2] for item in final_QS)
    ymean_fQS = wmean_y_fQS / sum(item[2] for item in final_QS)

    # beta_bar_num, beta_bar_denom = 0, 0
    beta_bar_num_fQS, beta_bar_denom_fQS, beta_bar_denom_fQS_d = 0, 0, 0

    # for i in range(len(vals)):
    #     beta_bar_num += ((x[i] - xmean) * (y[i] - ymean))
    # for i in range(len(vals)):
    #     beta_bar_denom += ((x[i] - xmean) ** 2)

    for x, y, val in final_QS:
        beta_bar_num_fQS += ((x - xmean_fQS) * (y - ymean_fQS))
    for x, y, val in final_QS:
        beta_bar_denom_fQS += ((x - xmean_fQS) ** 2)
    for x, y, val in final_QS:
        beta_bar_denom_fQS_d += ((y - ymean_fQS) ** 2)

    # beta_bar = beta_bar_num / beta_bar_denom
    # angle = np.arctan2(beta_bar_num, beta_bar_denom) / np.pi

    beta_bar_fQS = beta_bar_num_fQS / beta_bar_denom_fQS
    angle_fQS = np.arctan2(beta_bar_num_fQS, beta_bar_denom_fQS) / np.pi

    # beta_bar_fQS_d = beta_bar_denom_fQS_d / beta_bar_num_fQS
    # angle_fQS_d = np.arctan2(beta_bar_denom_fQS_d, beta_bar_num_fQS) / np.pi

    print("beta_bar related data in dirLR_ß:", [CX, CY], beta_bar_fQS, beta_bar_num_fQS, beta_bar_denom_fQS, xmean_fQS, ymean_fQS, angle_fQS)
    # print("beta_bar_d related data in dirLR_ß:", [CX, CY], beta_bar_fQS_d, beta_bar_num_fQS, beta_bar_denom_fQS_d, xmean_fQS, angle_fQS_d)

    points_possible = []
    points_possible_fQS = []

    print("values_coord in dirLR:", [CX, CY], values_coord)
    # print("fQS in dirLR:", [CX, CY], final_QS)

    # if beta_bar == 0:
    #     beta_bar = 10e-10

    if beta_bar_fQS == 0:
        beta_bar_fQS = 10e-7

    direction_list_temp_fQS = []
    direction_list_coord_fQS = []

    direction_general_fQS = None

    if ((angle_fQS >= 0.125) and (angle_fQS <= 0.375)) or \
            ((angle_fQS >= -0.875) and (angle_fQS <= -0.625)):
        direction_general_fQS = 'SE'
    elif ((angle_fQS >= 0.375) and (angle_fQS < 0.625)) or \
            ((angle_fQS > -0.625) and (angle_fQS < -0.375)):
        direction_general_fQS = 'S'
    elif ((angle_fQS >= 0.625) and (angle_fQS <= 0.875)) or \
            ((angle_fQS >= -0.375) and (angle_fQS <= -0.125)):
        direction_general_fQS = 'SW'
    elif ((angle_fQS >= 0) and (angle_fQS < 0.125)) or \
            ((angle_fQS >= -1) and (angle_fQS < -0.875)) or \
            ((angle_fQS > 0.875) and (angle_fQS <= 1)) or \
            ((angle_fQS > -0.125) and (angle_fQS < 0)):
        # ((angle < -0.125) and (angle > 0)):

        direction_general_fQS = 'E'

    if direction_general_fQS == 'E':
        beta_bar_fQS_perp = (-1) / beta_bar_fQS
        max_dist_point = [CX, CY]

        # for px, py in points_possible_fQS:
        for px, py, _ in final_QS:
            if (IMG_F[py][px] < LIMIT) and (np.linalg.norm(np.array([CX, CY]) - np.array([px, py])) >
                                            np.linalg.norm(np.array([CX, CY]) - np.array(max_dist_point))):
                max_dist_point = [px, py]

        a_l, b_l, c_l = beta_bar_fQS, -1, (beta_bar_fQS * CX) - CY
        d_l = abs(((a_l * max_dist_point[0]) + (b_l * max_dist_point[1]) - c_l)) / (np.sqrt((a_l ** 2) + (b_l ** 2)))

        a_pl, b_pl, c_pl = beta_bar_fQS_perp, -1, (beta_bar_fQS_perp * CX) - CY
        d_pl = abs(((a_pl * max_dist_point[0]) + (b_pl * max_dist_point[1]) - c_pl)) / (
            np.sqrt((a_pl ** 2) + (b_pl ** 2)))

        print("max_dist_point at:", [CX, CY], max_dist_point, beta_bar_fQS, beta_bar_fQS_perp)
        print("d_l and d_pl data at:", [CX, CY], d_l, d_pl)

        if d_pl < d_l:
            beta_bar_fQS = beta_bar_fQS_perp
            angle_fQS = np.arctan2((-1) * beta_bar_denom_fQS, beta_bar_num_fQS) / np.pi

            if ((angle_fQS >= 0.125) and (angle_fQS <= 0.375)) or \
                    ((angle_fQS >= -0.875) and (angle_fQS <= -0.625)):
                direction_general_fQS = 'SE'
            elif ((angle_fQS >= 0.375) and (angle_fQS < 0.625)) or \
                    ((angle_fQS > -0.625) and (angle_fQS < -0.375)):
                direction_general_fQS = 'S'
            elif ((angle_fQS >= 0.625) and (angle_fQS <= 0.875)) or \
                    ((angle_fQS >= -0.375) and (angle_fQS <= -0.125)):
                direction_general_fQS = 'SW'
            elif ((angle_fQS >= 0) and (angle_fQS < 0.125)) or \
                    ((angle_fQS >= -1) and (angle_fQS < -0.875)) or \
                    ((angle_fQS > 0.875) and (angle_fQS <= 1)) or \
                    ((angle_fQS > -0.125) and (angle_fQS < 0)):
                # ((angle < -0.125) and (angle > 0)):

                direction_general_fQS = 'E'

            print("correction applied at:", [CX, CY], direction_general_fQS)

    if True:
        for x, y, _ in final_QS:
            try:
                # print("error in divide by 0:", beta_bar, beta_bar_num, beta_bar_denom)

                lower_x = np.floor(((y - CY) / beta_bar_fQS) + CX)
                upper_x = lower_x + 1

                if (x == upper_x) or (x == lower_x):
                    points_possible.append([x, y])

                lower_y = np.floor(((x - CX) * beta_bar_fQS) + CY)
                upper_y = lower_y + 1

                if (y == upper_y) or (y == lower_y):
                    points_possible_fQS.append([x, y])

            except:
                print("error in divide by 0 in fQS:", beta_bar_fQS, beta_bar_num_fQS, beta_bar_denom_fQS)

    # direction_list = []
    degs = []

    print("points_possible in dirLR:", [CX, CY], points_possible)
    print("points_possible in dirLR fQS:", [CX, CY], points_possible_fQS)

    if True:
        # if PREV:
        #     PREV = opp[PREV]

        print("inside length > 0 with points_possible_fQS:", points_possible_fQS, direction_general_fQS, angle_fQS,
              beta_bar_fQS)

        # extremes = [points_possible[0], points_possible[-1]]
        # points_possible = []

        # for i in range(2):
        for curr in points_possible_fQS:
            # curr = extremes[i]
            prev = [CX, CY]

            delta_x = curr[0] - prev[0]
            delta_y = curr[1] - prev[1]
            theta_radians = np.arctan2(delta_y, delta_x)

            angle_new_format = theta_radians / np.pi

            # if (curr[0] != X) and (prev[0] != X) and (curr[1] != Y) and (prev[1] != Y):
            if True or ((curr[0] != XO) and (prev[0] != XO) and (curr[1] != YO) and (prev[1] != YO)):
                direction_pointing = None

                if ((angle_new_format >= 0.125) and (angle_new_format <= 0.375)) or \
                        ((angle_new_format >= -0.875) and (angle_new_format <= -0.625)):
                    direction_pointing = 'SE'
                elif ((angle_new_format >= 0.375) and (angle_new_format < 0.625)) or \
                        ((angle_new_format > -0.625) and (angle_new_format < -0.375)):
                    direction_pointing = 'S'
                elif ((angle_new_format >= 0.625) and (angle_new_format <= 0.875)) or \
                        ((angle_new_format >= -0.375) and (angle_new_format <= -0.125)):
                    direction_pointing = 'SW'
                elif ((angle_new_format >= 0) and (angle_new_format < 0.125)) or \
                        ((angle_new_format >= -1) and (angle_new_format < -0.875)) or \
                        ((angle_new_format > 0.875) and (angle_new_format <= 1)) or \
                        ((angle_new_format > -0.125) and (angle_new_format < 0)):
                    # ((angle_new_format < -0.125) and (angle_new_format > 0)):
                    direction_pointing = 'E'

                # print('direction_pointing before:', direction_pointing)

                if direction_pointing == direction_general_fQS:
                    # if True:

                    if direction_pointing == 'SE':
                        if (curr[1] <= CY):
                            direction_pointing = 'NW'
                    elif direction_pointing == 'S':
                        if (curr[1] <= CY):
                            direction_pointing = 'N'
                    elif direction_pointing == 'SW':
                        if (curr[1] <= CY):
                            direction_pointing = 'NE'
                    elif direction_pointing == 'E':
                        if (curr[0] <= CX):
                            direction_pointing = 'W'

                    # if PREV and (direction_pointing == PREV):
                    #     continue

                    # if direction_pointing in direction_list:
                    if True:
                        # points_possible.append(curr)

                        if direction_pointing not in direction_list_temp_fQS:
                            direction_list_temp_fQS.append(direction_pointing)
                            direction_list_coord_fQS.append(curr)
                            # direction_list_coord[direction_pointing].append(curr)

                        else:
                            try:
                                index_direction_list_temp = direction_list_temp_fQS.index(direction_pointing)

                                curr_temp = direction_list_coord_fQS[index_direction_list_temp]
                                # curr_temp = direction_list_coord[direction_pointing]
                                dist_temp = np.sqrt(((curr_temp[0] - CX) ** 2) + ((curr_temp[1] - CY) ** 2))
                                dist_curr = np.sqrt(((curr[0] - CX) ** 2) + ((curr[1] - CY) ** 2))

                                if dist_temp < dist_curr:
                                    direction_list_coord_fQS[index_direction_list_temp] = curr
                                    # direction_list_coord[direction_pointing] = curr

                            except:
                                print("error in direction_list_coord_fQS:", direction_list_coord_fQS)

                        # print("direction_list_coord_fQS in dirLR:", direction_pointing, curr,
                        #       [CX, CY], direction_list_coord_fQS)

        points_possible_fQS = direction_list_coord_fQS

    # print("points_possible, direction_list from dirLR at:", [CX, CY], direction_list, direction_list_temp, points_possible, flag_COI, direction_general, angle)
    print("points_possible_fQS, direction_list_fQS from dirLR at:", [CX, CY], direction_list_temp_fQS,
          points_possible_fQS, flag_COI,
          direction_general_fQS, angle_fQS, beta_bar_num_fQS, beta_bar_denom_fQS)

    # ----------------------------------------------

    final_point_fQS = []
    final_direction_fQS = []

    if True:
        print("points_possible_fQS from dirLR:", [CX, CY], PREV, points_possible_fQS, direction_list_temp_fQS)

        # if len(points_possible_fQS) != 0:
        if True:
            for i in range(len(direction_list_temp_fQS)):
                direction = direction_list_temp_fQS[i]
                if PREV and (direction != opp[PREV]):
                    final_direction_fQS.append(direction)

                    final_point_fQS.append(find_max_perp(points_possible_fQS[i], beta_bar_fQS, IMG_F, LIMIT))
                    # final_point_fQS.append(points_possible_fQS[i])

                elif not PREV:
                    final_direction_fQS = direction_list_temp_fQS
                    # final_point_fQS = points_possible_fQS
                    final_point_fQS = []

                    for point in points_possible_fQS:
                        final_point_fQS.append(find_max_perp(point, beta_bar_fQS, IMG_F, LIMIT))

            print("final data from dirLR at fQS:", [CX, CY], final_direction_fQS, final_point_fQS, beta_bar_fQS)

        # else:
        #     return beta_bar_fQS, angle_fQS, None, find_max_perp([CX, CY], beta_bar_fQS, IMG_F, LIMIT)

    # ----------------------------------------------

    # # ßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßß
    #
    # for dp in values_coord:
    #     if dp not in final_point_fQS:
    #         maintainer_matrix[dp[1]][dp[0]] = "d"
    #
    # final_point_t = []
    # final_direction_fQS_t = []
    #
    # for fp_x, fp_y in final_point_fQS:
    #     if maintainer_matrix[fp_y][fp_x] != "d":
    #         final_point_t.append([fp_x, fp_y])
    #         final_direction_fQS_t.append(final_direction_fQS[final_point_fQS.index([fp_x, fp_y])])
    #
    # final_point_fQS = final_point_t
    # final_direction_fQS = final_direction_fQS_t
    #
    # # ßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßß

    # ðððððððððððððððððððððððððððððððððððððððððððððð

    final_direction_fQS_t, final_point_fQS_t = [], []
    final_direction_fQS_tc, final_point_fQS_tc = [], []

    if PREV:
        for i in range(len(final_direction_fQS)):
            if PREV not in opp2[final_direction_fQS[i]]:
                final_direction_fQS_t.append(final_direction_fQS[i])
                final_point_fQS_t.append(final_point_fQS[i])

        final_direction_fQS = final_direction_fQS_t
        final_point_fQS = final_point_fQS_t

    print("PREV_COORDS data from dirLR_ß at:", [CX, CY], PREV_COORDS)

    if PREV_COORDS:
        for i in range(len(final_direction_fQS)):
            coord = final_point_fQS[i]

            if np.linalg.norm(np.array(coord) - np.array(PREV_COORDS)) > \
                np.linalg.norm(np.array([CX, CY]) - np.array(PREV_COORDS)):
                final_direction_fQS_tc.append(final_direction_fQS[i])
                final_point_fQS_tc.append(coord)

        final_direction_fQS = final_direction_fQS_tc
        final_point_fQS = final_point_fQS_tc

    # for i in range(len(final_point_fQS)):
    #     final_point_fQS[i] = find_max_perp(final_point_fQS[i], beta_bar_fQS, IMG_F, LIMIT)

    # ðððððððððððððððððððððððððððððððððððððððððððððð

    # ƒƒƒƒƒƒƒƒƒƒƒƒƒƒƒƒƒƒƒƒƒƒƒƒƒƒƒƒƒƒƒƒƒƒƒƒƒƒƒƒƒƒƒƒƒƒ

    for x, y in final_point_fQS:
        maintainer_matrix[y][x] = 0

    # ƒƒƒƒƒƒƒƒƒƒƒƒƒƒƒƒƒƒƒƒƒƒƒƒƒƒƒƒƒƒƒƒƒƒƒƒƒƒƒƒƒƒƒƒƒƒ

    end_dirLR = timer()
    print("time taken for dirLR_ß:", end_dirLR - start_dirLR)

    # return beta_bar, angle, direction, None
    # return beta_bar, angle, final_direction, final_point
    return beta_bar_fQS, angle_fQS, final_direction_fQS, final_point_fQS


terminal_coords = {}
Harris_corners_directions = {}
Harris_corners_angles = {}

start_terminal = timer()

for key in coords_for_COI:
    # start_sub_terminal = timer()
    coord = coords_for_COI[key]

    _, __, ___, X1, Y1 = grapher_3D(13, image_final, coord[0], coord[1])
    directions, pp, len_conts, angle = get_dirs_from_coords(_, __, ___, X1, Y1, None)

    # beta_bar_testing, angle_testing, direction_testing, point_testing = dirLR(image_final, X1, Y1, 27, 100)

    # print("directions in coords_for_COI loop:", directions, coord, len_conts)

    Harris_corners_directions[key] = directions
    Harris_corners_angles[key] = angle

    if len_conts == 1:
        terminal_coords[key] = coord

    # end_sub_terminal = timer()
    # print("time taken for sub-terminal to run:", end_sub_terminal - start_sub_terminal)

print("terminal_coords:", terminal_coords)
print("Harris_corners_directions:", Harris_corners_directions)

end_terminal = timer()

# # _, __, ___, X1, Y1 = grapher_3D(13, image_final, 497, 244)
# _, __, ___, X1, Y1 = grapher_3D(13, image_final, 538, 164)
# directions, pp, ____, angle = get_dirs_from_coords(_, __, ___, X1, Y1, None)
# print("data from manual coordinates 1:", directions, pp, ____, angle)
#
# directions, prob_arr, n = kernel_determiner(image_final, 497, 244, 0.46, 0.3581)
# print("data from manual coordinates 2:", directions, prob_arr, n)


# directions, prob_arr, n = kernel_determiner(image_final, 306, 273, 1.1, 0.901)
# print("direction, prob_arr, n at:", directions, prob_arr, n, [306, 273])
# directions, prob_arr, n = kernel_determiner(image_final, 305, 274, 1.1, 0.901)
# print("direction, prob_arr, n at:", directions, prob_arr, n, [305, 274])
# directions, prob_arr, n = kernel_determiner(image_final, 305, 272, 1.1, 0.901)
# print("direction, prob_arr, n at:", directions, prob_arr, n, [305, 272])
# directions, prob_arr, n = kernel_determiner(image_final, 304, 273, 1.1, 0.901)
# print("direction, prob_arr, n at:", directions, prob_arr, n, [304, 273])
# directions, prob_arr, n = kernel_determiner(image_final, 305, 274, 1.1, 0.901)
# print("direction, prob_arr, n at:", directions, prob_arr, n, [305, 274])
# directions, prob_arr, n = kernel_determiner(image_final, 304, 275, 1.1, 0.901)
# print("direction, prob_arr, n at:", directions, prob_arr, n, [304, 275])
# directions, prob_arr, n = kernel_determiner(image_final, 302, 275, 1.1, 0.901)
# print("direction, prob_arr, n at:", directions, prob_arr, n, [302, 275])

# n, prob_arr, directions = kernel_determiner(image_final, X, Y, 1.1, 0.901)
# print("manual directions, prob_arr:", directions, prob_arr, [X, Y], n)

ans_final = []

plt.figure("image_final")
plt.imshow(image_final)


# def mainFunction(X, Y, PREV, MODE, DIRECTIONS, POINT_NEXT=None):
#     # def mainFunction(X, Y, DIRECTIONS_INPUT, PROB_ARR_INPUT):
#     #     prev = None
#     prev = PREV
#
#     directions = DIRECTIONS
#
#     _, __, ___, X1, Y1 = grapher_3D(13, image_final, X, Y)
#
#     # if MODE == 1:
#     #     n, prob_arr, directions = kernel_determiner(image_final, X, Y, 1.1, 0.901)
#     #     print("directions MODE == 1", directions, [X, Y], prob_arr, n)
#
#     # if len(directions) > 2:
#     #     _, __, ___, X1, Y1 = grapher_3D(13, image_final, X, Y)
#     #     directions, pp, len_conts, angle = get_dirs_from_coords(_, __, ___, X1, Y1, None)
#     #
#     #     print("directions after len(directions) > 2 MODE == 1", directions, [X, Y], [X1, Y1])
#
#     if len(directions) > 2:
#         # _, __, ___, X1, Y1 = grapher_3D(13, image_final, X, Y)
#
#         terminal_key = None
#         for key in terminal_coords:
#             if terminal_key == None:
#                 if terminal_coords[key] == [X1, Y1]:
#                     terminal_key = key
#
#         if terminal_key != None:
#             n, prob_arr, directions = kernel_determiner(image_final, X1, Y1, 1.1, 0.901)
#             beta_bar_testing, angle_testing, direction_testing, point_testing = dirLR(image_final, X1, Y1, 27, 100)
#             print("directions because terminal point:", directions, direction_testing, [X, Y], [X1, Y1])
#         else:
#             print("directions after len(directions) > 2 MODE == 1", directions, [X, Y], [X1, Y1])
#
#         # directions, pp, len_conts, angle = get_dirs_from_coords(_, __, ___, X1, Y1, None)
#
#     # if prev and opp[prev] in directions:
#     #     directions.remove(opp[prev])
#
#     # _, __, ___, X1, Y1 = grapher_3D(13, image_final, X, Y)
#
#     # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#     # directions_temp, pp, len_conts, angle = get_dirs_from_coords(_, __, ___, X1, Y1, None)
#     #
#     # beta_bar_testing, angle_testing, direction_testing, point_testing = dirLR(image_final, X1, Y1, 27, 100)
#     # print("entry level data using grapher_3D:", directions_temp, [X1, Y1], [X, Y])
#     #
#     # print("entry level data:", directions, [X, Y], prev)
#     # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#
#     # else:
#     #     directions = Harris_corners_directions[[X, Y]]
#
#     # directions, prob_arr, n = kernel_determiner(image_final, X, Y, 0.46, 0.3581)
#     #
#     # print("directions, prob_arr:", directions, prob_arr, [X, Y], n)
#
#     # directions_T, prob_arr_T, n = kernel_determiner(image_final, X, Y, 1.1, -0.1)
#     #
#     # print("directions_T, prob_arr_T:", directions_T, prob_arr_T, [X, Y])
#
#     # DIRECTIONS = None
#
#     count = 0
#     temp = None
#
#     ans = ""
#
#     while len(directions) != 0:
#
#         if len(directions) == 1:
#             # print("len(directions) = 1 at:", [X, Y], directions)
#             if (not prev) or (directions[0] != opp[prev]):
#                 # if not prev:
#                 prev = directions[0]
#
#                 if not temp:
#                     temp = directions[0]
#                     count += 1
#                 elif temp == directions[0]:
#                     if count < resolution:
#                         count += 1
#                     else:
#                         # ans += "$" + str([X, Y]) + directions[0]
#                         ans += "$" + directions[0]
#                         count = 0
#                         temp = None
#                 else:
#                     count = 0
#                     temp = directions[0]
#
#                 print("data at each point:", [X, Y], directions)
#
#                 Y, X = kernel_operator(directions[0], Y, X, 1)
#                 # directions, prob_arr, n = kernel_determiner(image_final, X, Y, 0.46, 0.3581)
#                 n, prob_arr, directions = kernel_determiner(image_final, X, Y, 1.1, 0.901)
#
#                 try:
#                     beta_bar_testing, angle_testing, direction_testing, point_testing = dirLR(image_final, X, Y, 27, 100)
#
#                     print("direction_testing at", [X, Y], direction_testing, directions)
#
#                 except Exception as e:
#                     print("error in dirLR at line 2341:", [X, Y], e, directions)
#
#                 # if POINT_NEXT is None:
#                 #     Y, X = kernel_operator(directions[0], Y, X, 1)
#                 #     # directions, prob_arr, n = kernel_determiner(image_final, X, Y, 0.46, 0.3581)
#                 #     n, prob_arr, directions = kernel_determiner(image_final, X, Y, 1.1, 0.901)
#                 #     beta_bar_testing, angle_testing, direction_testing, point_testing = dirLR(image_final, X, Y, 27, 100)
#                 #
#                 #     print("direction_testing at", [X, Y], direction_testing, directions)
#                 #
#                 # else:
#                 #     X, Y = POINT_NEXT
#
#             elif directions[0] == opp[prev]:
#                 _, __, ___, XX, YY = grapher_3D(13, image_final, X, Y)
#                 print("terminating now at:", [X, Y], [XX, YY], directions)
#
#                 key = None
#
#                 for k in terminal_coords:
#                     if terminal_coords[k] == [XX, YY]:
#                         key = k
#
#                 print("key at len(directions) = 1 if terminal:", key)
#
#                 if not key:
#                     for k in coords_for_COI:
#                         if coords_for_COI[k] == [XX, YY]:
#                             key = k
#
#                     print("key at len(directions) = 1 if not terminal but COI:", key)
#
#                     if key:
#                         directions = Harris_corners_directions[key]
#                         # X, Y = recenter([X, Y], [XX, YY], Harris_corners_angles[[XX, YY]], 1, None, None, None)
#                         # n, prob_arr, directions = kernel_determiner(image_final, X, Y, 1.1, 0.901)
#                         directions.remove(opp[prev])
#                     else:
#                         # # X, Y = recenter([X, Y], [XX, YY], 0.99, 1, directions, directions, 1)
#                         # X, Y = recenter([X, Y], [XX, YY], 0.99, 1, [opp[directions[0]]], [opp[directions[0]]], 1)
#                         #
#                         # print("[X, Y] after recenter-ing from not COI not terminal:", [X, Y])
#
#                         # TODO: marker start
#
#                         directions, pp, ____, angle = get_dirs_from_coords(_, __, ___, X, Y, prev)
#
#                         beta_bar_testing, angle_testing, direction_testing, point_testing = dirLR(image_final, X, Y, 27,
#                                                                                                   100)
#                         print("directions - 1666:", [X, Y], directions, pp, ____, angle)
#
#                         X, Y = recenter([X, Y], [XX, YY], angle, 1, directions, directions, 0)
#                         # n, prob_arr, directions = kernel_determiner(image_final, X, Y, 1.1, 0.901)
#                         print("directions at [X, Y] - 1666:", directions, [X, Y])
#
#                         # TODO: marker end
#
#                         # TODO: marker 2 start
#
#                         # _, __, ___, X1, Y1 = grapher_3D(13, image_final, X, Y)
#                         #
#                         # tk = None
#                         #
#                         # for k in coords_for_COI:
#                         #     # tk = None
#                         #     if coords_for_COI[k] == [X1, Y1]:
#                         #         tk = k
#                         #
#                         # if tk:
#                         #     # if tk in Harris_corners_directions:
#                         #     directions = Harris_corners_directions[tk]
#                         #     angle = Harris_corners_angles[tk]
#                         # else:
#                         #     directions, pp, ____, angle = get_dirs_from_coords(_, __, ___, X1, Y1, prev)
#                         # # This directions is after removing the previous direction
#                         #
#                         # print("prev at each point:", [X, Y], prev)
#                         #
#                         # if opp[prev] in directions:
#                         #     directions.remove(opp[prev])
#                         #
#                         # print("directions after changed code in marker 2:", [X, Y], [X1, Y1], directions)
#
#                         # TODO: marker 2 end
#
#                         # n, prob_arr, directions = kernel_determiner(image_final, X, Y, 1.1, 0.901)
#                         #
#                         # # if opp[prev] in directions:
#                         # #     directions.remove(opp[prev])
#                         # #
#                         # print("directions after [X, Y] recenter-ing:", directions)
#                         # # # Y, X = kernel_operator(opp[directions[0]], Y, X, 1)
#                         # # # print("key is none at 1498, breaking:", [X, Y], directions)
#                         # # break
#                 else:
#                     print("terminal point encountered at:", [X, Y])
#                     break
#             else:
#                 print("broke at an overlooked part:", [X, Y], directions)
#                 break
#
#             print("after data at each point, next point and directions:", [X, Y], directions)
#
#         elif len(directions) > 1:
#             # if opp[prev] in directions:
#             if (prev is not None) and (opp[prev] in directions):
#                 directions.remove(opp[prev])
#
#             # print("directions after removing the opposite:", directions, prob_arr, [X, Y])
#
#             directions_comparison = directions.copy()
#             # directions_comparison is a copy of the directions array which contains only the next direction
#
#             if len(directions) == 1:
#                 if not temp:
#                     temp = directions[0]
#                     count += 1
#                 elif temp == directions[0]:
#                     if count < resolution:
#                         count += 1
#                     else:
#                         # ans += "$" + str([X, Y]) + directions[0]
#                         ans += "$" + directions[0]
#                         count = 0
#                         temp = None
#                 else:
#                     count = 0
#                     temp = directions[0]
#
#                 print("data at each point:", [X, Y], directions)
#
#                 Y, X = kernel_operator(directions[0], Y, X, 1)
#                 # directions, prob_arr, n = kernel_determiner(image_final, X, Y, 0.46, 0.3581)
#                 n, prob_arr, directions = kernel_determiner(image_final, X, Y, 1.1, 0.901)
#                 beta_bar_testing, angle_testing, direction_testing, point_testing = dirLR(image_final, X, Y, 27, 100)
#
#                 print("direcion_testing at:", [X, Y], direction_testing, directions)
#             elif len(directions) > 1:
#                 try:
#                     # if True:
#                     _, __, ___, X1, Y1 = grapher_3D(13, image_final, X, Y)
#                     print("len(directions) > 1 at:", [X, Y], [X1, Y1], directions)
#
#                     # if ((X1, Y1) not in centers_traversing) or True:
#                     if (X1, Y1) not in centers_traversing:
#                         # if (X1, Y1) not in centers_traversing:
#                         #     centers_traversing[(X1, Y1)] = '-'
#
#                         print("coord not in centers_traversing yet:", [X1, Y1])
#
#                         centers_traversing[(X1, Y1)] = '-'
#
#                         directions, angle = None, None
#
#                         tk = None
#
#                         for k in coords_for_COI:
#                             # tk = None
#                             if coords_for_COI[k] == [X1, Y1]:
#                                 tk = k
#
#                         if tk:
#                             # if tk in Harris_corners_directions:
#                             directions = Harris_corners_directions[tk]
#                             angle = Harris_corners_angles[tk]
#                         else:
#                             directions, pp, ____, angle = get_dirs_from_coords(_, __, ___, X1, Y1, prev)
#                             beta_bar_testing, angle_testing, direction_testing, point_testing = dirLR(image_final, X1,
#                                                                                                       Y1, 27, 100)
#                         # This directions is after removing the previous direction
#
#                         print("prev at each point:", [X, Y], prev)
#
#                         if opp[prev] in directions:
#                             directions.remove(opp[prev])
#
#                         if opp[prev] in directions_comparison:
#                             directions_comparison.remove(opp[prev])
#
#                         print("new directions at len(directions) > 1 at:", [X, Y], [X1, Y1], directions)
#
#                         if len(directions) == 1:
#                             # recenter function comes here
#                             if not temp:
#                                 temp = directions[0]
#                                 count += 1
#                             elif temp == directions[0]:
#                                 if count < resolution:
#                                     count += 1
#                                 else:
#                                     # ans += "$" + str([X, Y]) + directions[0]
#                                     ans += "$" + directions[0]
#                                     count = 0
#                                     temp = None
#                             else:
#                                 count = 0
#                                 temp = directions[0]
#
#                             print("data at each point:", [X, Y], directions)
#
#                             # X, Y = recenter([X, Y], [X1, Y1], angle, 1, directions, directions_comparison, prev)
#
#                             X, Y = recenter([X, Y], [X1, Y1], angle, 1, directions, directions, 0)
#                             # Y, X = recenter([X, Y], [X1, Y1], angle, 1, directions, directions, 0)
#
#                             n, prob_arr, directions = kernel_determiner(image_final, X, Y, 1.1, 0.901)
#                             beta_bar_testing, angle_testing, direction_testing, point_testing = dirLR(image_final, X, Y,
#                                                                                                       27, 100)
#
#                             print("direction_testing at:", [X, Y], direction_testing, directions)
#                             # pass
#                         else:
#                             # loop for multiple directions comes here
#                             print("printed multiple directions at:", [X, Y], [X1, Y1], directions)
#                             ans += "$$"
#
#                             for i in range(len(directions)):
#                                 Y_Temp, X_Temp = kernel_operator(directions[i], Y1, X1,
#                                                                  getstep(directions[i], X1, Y1, dict_for_coords))
#
#                                 print("some data:", [X_Temp, Y_Temp], [X1, Y1], directions)
#                                 # print("some data:", [X1, Y1], [X1, Y1], directions)
#                                 mainFunction(X_Temp, Y_Temp, directions[i], 2, directions[i])
#                                 # mainFunction(X1, Y1, directions[i], 2, directions)
#
#                                 # try:
#                                 #     Y_Temp, X_Temp = kernel_operator(directions[i], Y1, X1, getstep(directions[i], X1, Y1, dict_for_coords))
#                                 #
#                                 #     print("some data:", [X_Temp, Y_Temp], [X1, Y1], directions)
#                                 #     # print("some data:", [X1, Y1], [X1, Y1], directions)
#                                 #     mainFunction(X_Temp, Y_Temp, directions[i], 2, directions[i])
#                                 #     # mainFunction(X1, Y1, directions[i], 2, directions)
#                                 # except:
#                                 #     print("error in the for loop at:", [X, Y], directions[i])
#                             # pass
#                     else:
#                         print("something has happened at:", [X, Y], [X1, Y1], directions, prev)
#                         break
#                 except Exception as error:
#                     print("error occurred at len(directions) > 1 at:", [X, Y], str(error), prev)
#                     break
#
#                 # TODO: Just a marker
#
#                 # # if ((X1, Y1) not in centers_traversing) or True:
#                 # if (X1, Y1) not in centers_traversing:
#                 #     # if (X1, Y1) not in centers_traversing:
#                 #     #     centers_traversing[(X1, Y1)] = '-'
#                 #
#                 #     centers_traversing[(X1, Y1)] = '-'
#                 #
#                 #     if len(directions) > 1:
#                 #         # print("printed multiple directions at:", [X, Y], [X1, Y1], directions, pp)
#                 #         print("printed multiple directions at:", [X, Y], [X1, Y1], directions)
#                 #         ans += "$$"
#                 #         # print("break at 1477")
#                 #
#                 #         for i in range(len(directions)):
#                 #             # X_Temp, Y_Temp = kernel_operator(directions[0], Y1, X1, 6)
#                 #             # Y_Temp, X_Temp = kernel_operator(directions[0], Y1, X1, 9)
#                 #             # Y_Temp, X_Temp = kernel_operator(directions[i], Y1, X1, 9)
#                 #             Y_Temp, X_Temp = kernel_operator(directions[i], Y1, X1, getstep(directions[i], X1, Y1, dict_for_coords))
#                 #
#                 #             print("some data:", [X_Temp, Y_Temp], [X1, Y1], directions)
#                 #
#                 #             # mainFunction(X_Temp, Y_Temp, opp[directions[0]], None)
#                 #             # mainFunction(X_Temp, Y_Temp, directions[0], None)
#                 #             # mainFunction(X_Temp, Y_Temp, directions[0], None)
#                 #             mainFunction(X_Temp, Y_Temp, directions[i], None)
#                 #         break
#                 #     else: # 'else' is for the case where there is only one direction more to go
#                 #         print("directions:", directions, directions_comparison, [X, Y])
#                 #         X, Y = recenter([X, Y], [X1, Y1], angle, 1, directions, directions_comparison, pp)
#                 #
#                 #         n, prob_arr, directions = kernel_determiner(image_final, X, Y, 1.1, 0.91)
#                 #
#                 #         print("new directions from else block:", directions, n, [X, Y])
#                 #         # print("break at 1487")
#                 #
#                 #         # if not temp:
#                 #         #     temp = directions[0]
#                 #         #     count += 1
#                 #         # elif temp == directions[0]:
#                 #         #     if count < resolution:
#                 #         #         count += 1
#                 #         #     else:
#                 #         #         # ans += "$" + str([X, Y]) + directions[0]
#                 #         #         ans += "$" + directions[0]
#                 #         #         count = 0
#                 #         #         temp = None
#                 #         # else:
#                 #         #     count = 0
#                 #         #     temp = directions[0]
#                 #         #
#                 #         # Y, X = kernel_operator(directions[0], Y, X, 1)
#                 #         # # directions, prob_arr, n = kernel_determiner(image_final, X, Y, 0.46, 0.3581)
#                 #         # n, prob_arr, directions = kernel_determiner(image_final, X, Y, 1.1, 0.91)
#                 #         #
#                 #         # print("new directions from else block:", directions, n)
#                 #
#                 #         # break
#                 #
#                 # # TODO: Uncomment from here for the remaining else block
#                 # else:
#                 #     print("an error occurred in traversing the points at:", [X1, Y1], [X, Y], directions)
#                 #
#                 #     break
#             else:
#                 print("len(directions) == 0 at:", [X, Y])
#                 break
#     else:
#         print("len(directions) = 0:", directions, [X, Y])
#
#     if ans:
#         print("ans_ans_ans:", ans, "end of an ans segment", [X, Y], directions)
#
#     if ans != '':
#         # print("ans:", ans, "end of an ans segment", [X, Y], directions)
#         ans_final.append(ans)

def mainFunction2(X, Y, PREV, DIRECTIONS, POINT_NEXT=None, prev_coords=None):
    if PREV and opp[PREV] in DIRECTIONS:
        DIRECTIONS.remove(opp[PREV])

    print(Fore.GREEN + "ans_ans_ans:", [X, Y], PREV, DIRECTIONS, Style.RESET_ALL)

    temp_str = str(X) + ", " + str(Y)

    ANS[temp_str] = DIRECTIONS

    if len(DIRECTIONS) == 0:
        flag_terminal_coord = False

        for key in terminal_coords:
            if (flag_terminal_coord is False) and (terminal_coords[key] == [X, Y]):
                flag_terminal_coord = True

        if flag_terminal_coord is True:
            print("end of current iteration at:", [X, Y])

        else:
            print("false end detected at:", [X, Y])

    elif len(DIRECTIONS) == 1:
        # beta_bar_testing, angle_testing, directions_testing, points_testing = dirLR(image_final, X, Y, 13, 100, PREV)
        beta_bar_testing, angle_testing, directions_testing, points_testing = dirLR_β(image_final, X, Y, 27, 100, PREV,
                                                                                      PREV_COORDS=prev_coords)

        print("data len(DIRECTIONS) == 1 at:", [X, Y], directions_testing, points_testing, PREV, opp[PREV] if PREV is not None else "None")

        # mainFunction2(points_testing[0][0], points_testing[0][1], DIRECTIONS[0], [directions_testing[0]])

        # if angle_testing:
        if True:
            if beta_bar_testing:
                index = None
                for i in range(len(directions_testing)):
                    if PREV and (directions_testing[i] == opp[PREV]):
                        index = i
                        break

                if index is not None:
                    directions_testing.remove(directions_testing[index])
                    points_testing.remove(points_testing[index])

                # mainFunction2(points_testing[0][0], points_testing[0][1], DIRECTIONS[0], [directions_testing[0]])

                if len(directions_testing) > 1:
                    print("len(directions_testing) > 1 some error occurred in dirLR at:", [X, Y], directions_testing)

                elif len(directions_testing) == 1:
                    mainFunction2(points_testing[0][0], points_testing[0][1], DIRECTIONS[0], [directions_testing[0]], prev_coords=[X, Y])

                else:
                    print("len(directions_testing) == 0 some error occurred in dirLR at:", [X, Y], points_testing)
                    # print("relocated point at:", [X, Y], "is:", find_max_perp([X, Y], -0.9664187145904831, image_final, 100))

                    X_Reloc, Y_Reloc = find_max_perp([X, Y], beta_bar_testing, image_final, 100)
                    print("relocated point at:", [X, Y], "is:", [X_Reloc, Y_Reloc])

                    beta_bar_testing, angle_testing, directions_testing, points_testing = \
                        dirLR_β(image_final, X_Reloc, Y_Reloc, 27, 100, PREV, PREV_COORDS=prev_coords)

                    mainFunction2(points_testing[0][0], points_testing[0][1], DIRECTIONS[0], [directions_testing[0]], prev_coords=[X, Y])

            else:
                _, __, ___, X1, Y1 = grapher_3D(13, image_final, points_testing[0], points_testing[1])
                directions, pp, len_conts, angle = get_dirs_from_coords(_, __, ___, X1, Y1, None)

                if PREV and (opp[PREV] in directions):
                    directions.remove(opp[PREV])

                print("multiple directions here at:", points_testing, directions)

                for i in range(len(directions)):
                    try:
                        Y_Temp, X_Temp = kernel_operator(directions[i], Y1, X1,
                                                         getstep(directions[i], X1, Y1, dict_for_coords))

                        print("data before applying dirLR_β at:", [X_Temp, Y_Temp], [X1, Y1], directions[i], "at harris:", points_testing)

                        # Y_Temp, X_Temp = kernel_operator(directions[i], points_testing[1], points_testing[0],
                        #                                  getstep(directions[i], points_testing[1], points_testing[0], dict_for_coords))
                        #
                        # print("data before applying dirLR_β at:", [X_Temp, Y_Temp], [X1, Y1], directions[i])

                        # beta_bar_testing, angle_testing, directions_testing, points_testing = \
                        #     dirLR(image_final, X_Temp, Y_Temp, 13, 100, directions[i])
                        beta_bar_testing, angle_testing, directions_testing, points_testing = \
                            dirLR_β(image_final, X_Temp, Y_Temp, 27, 100, directions[i], PREV_COORDS=prev_coords)

                        print("data inside for loop at:", [X_Temp, Y_Temp], [X, Y], directions_testing, points_testing, directions[i],
                              opp[directions[i]] if directions[i] is not None else "None")

                        mainFunction2(X_Temp, Y_Temp, directions[i], [directions_testing[0]], prev_coords=[X, Y])

                    except Exception as e:
                        print("some error occurred at:", points_testing, [X1, Y1], e)
    else:
        pass


# def mainFunction(X, Y, PREV, MODE, DIRECTIONS, POINT_NEXT = None):
#     pass
##########

key = None
_, __, ___, X1, Y1 = grapher_3D(13, image_final, XO, YO)

# X1, Y1 = 201, 129
for k in coords_for_COI:
    if coords_for_COI[k] == [X1, Y1]:
        key = k

# print("123456789")
# print("before start:", key, Harris_corners_directions, Harris_corners_angles)
# print("before start:", Harris_corners_directions[key], Harris_corners_angles[key])
# print("123456789")

# X, Y = recenter([X, Y], [X1, Y1], Harris_corners_angles[key], 1, Harris_corners_directions[key], Harris_corners_directions[key], 0)
# # Y, X = recenter([X, Y], [X1, Y1], Harris_corners_angles[key], 1, Harris_corners_directions[key], Harris_corners_directions[key], 0)
# print("")

_, __, ___, X1, Y1 = grapher_3D(13, image_final, XO, YO)
directions, pp, ____, angle = get_dirs_from_coords(_, __, ___, X1, Y1, None)

beta_bar_testing, angle_testing, direction_testing, point_testing = dirLR(image_final, X1, Y1, 13, 100, None)

print("before starting:", [XO, YO], [X1, Y1], directions, direction_testing, point_testing)

# end1 = timer()

# mainFunction(X, Y, None, 1, None)
# mainFunction(X1, Y1, None, 1, [direction_testing], point_testing)
# mainFunction(X1, Y1, None, 1, directions, point_testing)

# mainFunction2(X1, Y1, None, directions, point_testing)
start2 = timer()
mainFunction2(X1, Y1, None, direction_testing, POINT_NEXT=point_testing, prev_coords=None)
end2 = timer()
print("time taken for mainFunction2:", end2 - start2)




# beta_bar_testing, angle_testing, direction_testing, point_testing = dirLR(image_final, 205, 132, 13, 100, 'SE')

# beta_bar_testing, angle_testing, direction_testing, point_testing = dirLR_β(image_final, 211, 136, 13, 100, None)
# Nonebeta_bar_testing, angle_testing, direction_testing, point_testing = dirLR_β(image_final, 502, 237, 27, 100, "NE")
# beta_bar_testing, angle_testing, direction_testing, point_testing = dirLR_β(image_final, 501, 233, 27, 100, "NE")
# beta_bar_testing, angle_testing, direction_testing, point_testing = dirLR_β(image_final, 568, 144, 27, 100, "NE")
# beta_bar_testing, angle_testing, direction_testing, point_testing = dirLR_β(image_final, 567, 142, 27, 100, "NE")

# beta_bar_testing, angle_testing, direction_testing, point_testing = dirLR_β(image_final, 500, 265, 27, 100, "NW")
# beta_bar_testing, angle_testing, direction_testing, point_testing = dirLR_β(image_final, 521, 183, 27, 100, "NE")

# beta_bar_testing, angle_testing, direction_testing, point_testing = dirLR_β(image_final, 748, 236, 27, 100, "SE")
# beta_bar_testing, angle_testing, direction_testing, point_testing = dirLR_β(image_final, 749, 268, 27, 100, "S")
# beta_bar_testing, angle_testing, direction_testing, point_testing = dirLR_β(image_final, 748, 271, 27, 100, "SW")
# beta_bar_testing, angle_testing, direction_testing, point_testing = dirLR_β(image_final, 521, 183, 27, 100, "NE")
# beta_bar_testing, angle_testing, direction_testing, point_testing = dirLR_β(image_final, 727, 325, 27, 100, "SW")

# beta_bar_testing, angle_testing, direction_testing, point_testing = dirLR_β(image_final, 502, 237, 27, 100, "NE")
# beta_bar_testing, angle_testing, direction_testing, point_testing = dirLR(image_final, 502, 237, 13, 100, None)

# beta_bar_testing, angle_testing, direction_testing, point_testing = dirLR(image_final, 501, 238, 27, 100)
# beta_bar_testing, angle_testing, direction_testing, point_testing = dirLR(image_final, 500, 265, 27, 100)
# beta_bar_testing, angle_testing, direction_testing, point_testing = dirLR(image_final, 749, 251, 27, 100)

# print("custom data:", find_max_perp([566, 144], -0.40851619644276577, image_final, 100))
# print("custom data:", find_max_perp([631, 380], -0.10198771317350785, image_final, 100))
# print("custom data:", find_max_perp([568, 144], -0.5290962998716804, image_final, 100))
# print("custom data:", find_max_perp([747, 257], -1.8715019555877688, image_final, 100))
print("custom data:", find_max_perp([725, 323], -0.9664187145904831, image_final, 100))

# beta_bar_testing, angle_testing, direction_testing, point_testing = dirLR(image_final, 507, 210, 13, 100, None)
# beta_bar_testing, angle_testing, direction_testing, point_testing = dirLR(image_final, 511, 205, 13, 100, None)
# beta_bar_testing, angle_testing, direction_testing, point_testing = dirLR(image_final, 509, 204, 13, 100, None)

# beta_bar_testing, angle_testing, direction_testing, point_testing = dirLR(image_final, 507, 210, 27, 100, None)
# beta_bar_testing, angle_testing, direction_testing, point_testing = dirLR(image_final, 199, 379, 27, 100, None)
# beta_bar_testing, angle_testing, direction_testing, point_testing = dirLR(image_final, 587, 134, 27, 127, None)
# beta_bar_testing, angle_testing, direction_testing, point_testing = dirLR(image_final, 610, 128, 27, 127, None)
# beta_bar_testing, angle_testing, direction_testing, point_testing = dirLR(image_final, 585, 133, 27, 127, None)

# beta_bar_testing, angle_testing, direction_testing, point_testing = dirLR(image_final, 501, 273, 27, 127, None)
# beta_bar_testing, angle_testing, direction_testing, point_testing = dirLR(image_final, 501, 273, 29, 127, "S")

# beta_bar_testing, angle_testing, direction_testing, point_testing = dirLR(image_final, 505, 223, 27, 127, None)
# beta_bar_testing, angle_testing, direction_testing, point_testing = dirLR(image_final, 502, 225, 27, 127, None)
# beta_bar_testing, angle_testing, direction_testing, point_testing = dirLR(image_final, 503, 232, 27, 127, None)

# print("custom starting:", [725, 323], direction_testing, point_testing)




# print("time taken before mainFunction2:", end1 - start1)

##########

# _, __, ___, X1, Y1 = grapher_3D(13, image_final, 400, 502)
# _, __, ___, X1, Y1 = grapher_3D(13, image_final, 453, 748)
# _, __, ___, X1, Y1 = grapher_3D(13, image_final, 454, 751)
# _, __, ___, X1, Y1 = grapher_3D(13, image_final, 466, 751)
# _, __, ___, X1, Y1 = grapher_3D(13, image_final, 495, 253)
# _, __, ___, X1, Y1 = grapher_3D(13, image_final, 452, 743)
# _, __, ___, X1, Y1 = grapher_3D(13, image_final, 452, 748)

# ##########
# _, __, ___, X1, Y1 = grapher_3D(13, image_final, 497, 242)
# directions, pp, ____, angle = get_dirs_from_coords(_, __, ___, X1, Y1, None)
# print("directions manual:", [X1, Y1], directions, pp, ____, angle)
#
# dirrs, prob_arr, __ = kernel_determiner(image_final, 466, 751, 1.1, -1.0)
# # __, prob_arr, directions = kernel_determiner(image_final, 466, 751, 1.1, -1.0)
# print("directions manual 2:", [466, 751], dirrs, prob_arr, __)
# ##########

# print("ans_final:", ans_final)

print(juncs)
print("centers:", centers)
print("centers_traversing:", centers_traversing)

plt.figure("image_final_Harris")
plt.imshow(image_final_Harris)

plt.figure("grayscaled")
plt.imshow(image_final)

values = {}
# for y in range(204, 217):
#     for x in range(501, 514):

# 222, 150
# for y in range(144, 157):
#     for x in range(216, 229):

# 412, 253
# for y in range(247, 260):
#     for x in range(406, 419):

# # 504, 220
# for y in range(214, 227):
#     for x in range(498, 511):
#         values[(x, y)] = image_final[y][x]
#         # print("value at:", [x, y], image_final[y][x])
#
# _, __, direction_testing, point_testing = dirLR(image_final, 504, 220, 27, 100)
# print("values:", direction_testing, point_testing, "-d-v-", values)

end = timer()
print("time taken for program to run:", end - start)
print("time taken for terminal loop to run:", end_terminal - start_terminal)

for y in range(len(maintainer_matrix)):
    for x in range(len(maintainer_matrix[0])):
        if type(maintainer_matrix[y][x]) == str:
            maintainer_matrix[y][x] = 0

# print("ANS:", ANS)
# print("coords_for_COI:", coords_for_COI)

# print("mm:", maintainer_matrix)
plt.figure("maintainer_matrix")
plt.imshow(maintainer_matrix)

plt.show()

cv.waitKey(0)
cv.destroyAllWindows()
