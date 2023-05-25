import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from scipy.spatial import distance
from sklearn.cluster import DBSCAN

from timeit import default_timer as timer

start = timer()

# D2I = None
dir_dis = {"N": np.array([0, -1]), "NE": np.array([+1, -1]), "E": np.array([+1, 0]), "SE": np.array([+1, +1]),
           "S": np.array([0, +1]), "SW": np.array([-1, +1]), "W": np.array([-1, 0]), "NW": np.array([-1, -1])}
dict_bool = {0: "N", 1: "NE", 2: "E", 3: "SE", 4: "S", 5: "SW", 6: "W", 7: "NW"}
dirs = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
opp = {"N": "S", "NE": "SW", "E": "W", "SE": "NW", "S": "N", "SW": "NE", "W": "E", "NW": "SE"}
basic = ["N", "E", "S", "W"]
juncs = {}

resolution = 1 # decides the resolution with which the hand movements will happen unless interrupted by direction change
ans = ""

centers_traversing = {}

image = cv.imread('test image.png')
# image = cv.imread('scythe.png')

avgBlur_thinner = cv.blur(image, (5, 5))
image_final_thinner = cv.cvtColor(avgBlur_thinner, cv.COLOR_BGR2GRAY)
image_final_thinner = np.array(image_final_thinner)
# cv.imshow("thinner", avgBlur_thinner)

avgBlur = cv.medianBlur(image, 5)
# avgBlur = cv.bilateralFilter(image, 7, 75, 75)

# cv.imshow("BLur:", avgBlur)

# X, Y = 118, 76
X, Y = 201, 129
# X, Y = 289, 146

image_final = cv.cvtColor(avgBlur, cv.COLOR_BGR2GRAY)
image_final_test = np.bitwise_not(image_final)
image_final_test = np.float32(image_final_test)
image_final = np.array(image_final)

dst = cv.cornerHarris(image_final_test, 2, 13, 0.05)

dst = cv.dilate(dst, None)
image_Harris = image.copy()
image_Harris[dst > 0.01 * dst.max()] = [0, 0, 255]

# image_final_Harris = image_final.copy()
image_final_Harris = np.zeros([len(image_final), len(image_final[0])])

coords = [[x, y] for y in range(len(image_Harris)) for x in range(len(image_Harris[0])) if
          (image_Harris[y][x][2] == 255) and (image_Harris[y][x][1] == 0) and (image_Harris[y][x][0] == 0)]
coords_final = [coord for coord in coords if image_final_Harris[coord[1]][coord[0]] < 200]

def HarrisCorners(COORDS_FINAL, IMAGE_FINAL):
    eps = 3

    distances = distance.cdist(COORDS_FINAL, COORDS_FINAL, 'euclidean')

    db = DBSCAN(eps=eps, min_samples=1, metric='precomputed').fit(distances)

    labels = db.labels_
    coords_for_groups = {}
    colors_for_groups = {}

    count = 0

    # Print the groups of close coordinates
    for label in set(labels):
        count += 1

        group = [coord for i, coord in enumerate(COORDS_FINAL) if labels[i] == label]

        colors_for_groups[label] = int(100 + count)
        # colors_for_groups[300 + count] = label

        coords_for_groups[label] = group

    for label in coords_for_groups.keys():
        coords_corners = coords_for_groups[label]

        for coord_corner in coords_corners:
            IMAGE_FINAL[coord_corner[1]][coord_corner[0]] = colors_for_groups[label]

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

    return yy, xx


def kernel_determiner(I, CX, CY, UB, LB):
    KERNEL2 = np.array([[I[CY - 5][CX - 5], I[CY - 5][CX - 4], I[CY - 5][CX - 3], I[CY - 5][CX - 2], I[CY - 5][CX - 1],
                         I[CY - 5][CX], I[CY - 5][CX + 1], I[CY - 5][CX + 2], I[CY - 5][CX + 3], I[CY - 5][CX + 4],
                         I[CY - 5][CX + 5]],
                        [I[CY - 4][CX - 5], I[CY - 4][CX - 4], I[CY - 4][CX - 3], I[CY - 4][CX - 2], I[CY - 4][CX - 1],
                         I[CY - 4][CX], I[CY - 4][CX + 1], I[CY - 4][CX + 2], I[CY - 4][CX + 3], I[CY - 4][CX + 4],
                         I[CY - 4][CX + 5]],
                        [I[CY - 3][CX - 5], I[CY - 3][CX - 4], I[CY - 3][CX - 3], I[CY - 3][CX - 2], I[CY - 3][CX - 1],
                         I[CY - 3][CX], I[CY - 3][CX + 1], I[CY - 3][CX + 2], I[CY - 3][CX + 3], I[CY - 3][CX + 4],
                         I[CY - 3][CX + 5]],
                        [I[CY - 2][CX - 5], I[CY - 2][CX - 4], I[CY - 2][CX - 3], I[CY - 2][CX - 2], I[CY - 2][CX - 1],
                         I[CY - 2][CX], I[CY - 2][CX + 1], I[CY - 2][CX + 2], I[CY - 2][CX + 3], I[CY - 2][CX + 4],
                         I[CY - 2][CX + 5]],
                        [I[CY - 1][CX - 5], I[CY - 1][CX - 4], I[CY - 1][CX - 3], I[CY - 1][CX - 2], I[CY - 1][CX - 1],
                         I[CY - 1][CX], I[CY - 1][CX + 1], I[CY - 1][CX + 2], I[CY - 1][CX + 3], I[CY - 1][CX + 4],
                         I[CY - 1][CX + 5]],
                        [I[CY][CX - 5], I[CY][CX - 4], I[CY][CX - 3], I[CY][CX - 2], I[CY][CX - 1], I[CY][CX],
                         I[CY][CX + 1], I[CY][CX + 2], I[CY][CX + 3], I[CY][CX + 4], I[CY][CX + 5]],
                        [I[CY + 1][CX - 5], I[CY + 1][CX - 4], I[CY + 1][CX - 3], I[CY + 1][CX - 2], I[CY + 1][CX - 1],
                         I[CY + 1][CX], I[CY + 1][CX + 1], I[CY + 1][CX + 2], I[CY + 1][CX + 3], I[CY + 1][CX + 4],
                         I[CY + 1][CX + 5]],
                        [I[CY + 2][CX - 5], I[CY + 2][CX - 4], I[CY + 2][CX - 3], I[CY + 2][CX - 2], I[CY + 2][CX - 1],
                         I[CY + 2][CX], I[CY + 2][CX + 1], I[CY + 2][CX + 2], I[CY + 2][CX + 3], I[CY + 2][CX + 4],
                         I[CY + 2][CX + 5]],
                        [I[CY + 3][CX - 5], I[CY + 3][CX - 4], I[CY + 3][CX - 3], I[CY + 3][CX - 2], I[CY + 3][CX - 1],
                         I[CY + 3][CX], I[CY + 3][CX + 1], I[CY + 3][CX + 2], I[CY + 3][CX + 3], I[CY + 3][CX + 4],
                         I[CY + 3][CX + 5]],
                        [I[CY + 4][CX - 5], I[CY + 4][CX - 4], I[CY + 4][CX - 3], I[CY + 4][CX - 2], I[CY + 4][CX - 1],
                         I[CY + 4][CX], I[CY + 4][CX + 1], I[CY + 4][CX + 2], I[CY + 4][CX + 3], I[CY + 4][CX + 4],
                         I[CY + 4][CX + 5]],
                        [I[CY + 5][CX - 5], I[CY + 5][CX - 4], I[CY + 5][CX - 3], I[CY + 5][CX - 2], I[CY + 5][CX - 1],
                         I[CY + 5][CX], I[CY + 5][CX + 1], I[CY + 5][CX + 2], I[CY + 5][CX + 3], I[CY + 5][CX + 4],
                         I[CY + 5][CX + 5]]])

    dirs_or2 = {
        "N": [[0, 3], [0, 4], [0, 5], [0, 6], [0, 7], [1, 4], [1, 5], [1, 6], [2, 4], [2, 5], [2, 6], [3, 4], [3, 5],
              [3, 6], [4, 5]],
        "NE": [[4, 6], [3, 7], [2, 7], [2, 8], [3, 8], [1, 7], [1, 8], [1, 9], [2, 9], [3, 9], [0, 8], [0, 9], [0, 10],
               [1, 10], [2, 10]],
        "E": [[5, 6], [4, 7], [5, 7], [6, 7], [4, 8], [5, 8], [6, 8], [4, 9], [5, 9], [6, 9], [3, 10], [4, 10], [5, 10],
              [6, 10], [7, 10]],
        "SE": [[6, 6], [7, 7], [8, 7], [8, 8], [7, 8], [9, 7], [9, 8], [9, 9], [8, 9], [7, 9], [10, 8], [10, 9],
               [10, 10], [9, 10], [8, 10]],
        "S": [[6, 5], [7, 4], [7, 5], [7, 6], [8, 4], [8, 5], [8, 6], [9, 4], [9, 5], [9, 6], [10, 3], [10, 4], [10, 5],
              [10, 6], [10, 7]],
        "SW": [[6, 4], [7, 3], [7, 2], [8, 3], [8, 3], [7, 1], [8, 1], [9, 1], [9, 2], [9, 3], [8, 0], [9, 0], [10, 0],
               [10, 1], [10, 2]],
        "W": [[5, 4], [4, 3], [5, 3], [6, 3], [4, 2], [5, 2], [6, 2], [4, 1], [5, 1], [6, 1], [3, 0], [4, 0], [5, 0],
              [6, 0], [7, 0]],
        "NW": [[4, 4], [3, 3], [3, 2], [2, 2], [2, 3], [3, 1], [2, 1], [1, 1], [1, 2], [1, 3], [2, 0], [1, 0], [0, 0],
               [0, 1], [0, 2]]}

    probs2 = np.array([np.average([255 - KERNEL2[coords2[0]][coords2[1]] for coords2 in dirs_or2[d2]]) / 255 for d2 in dirs_or2.keys()])
    # print("probs2:", probs2)

    # probs = np.array([np.sqrt(np.mean([KERNEL[coords[0]][coords[1]] ** 2 for coords in dirs_or[d]])) / 255 for d in dirs_or.keys()])
    # print(probs)

    # print(np.var(probs))
    # print(np.sqrt(np.mean(probs ** 2)))

    # ans2 = [dirs[index] for index in range(8) if probs[index] < THRESH]
    ans2 = [dirs[index2] for index2 in range(8) if (probs2[index2] < UB) and (probs2[index2] > LB)]
    # print("ans2:", ans2)

    probs2_final = group_numbers(probs2, 0.2)[-1]
    ans2_final = [dirs[index] for index in range(8) if probs2[index] in probs2_final]

    # TODO: To later uncomment for debugging # print("ans2, probs2 from kernel_determiner:", [CX, CY], ans2, probs2)

    # print("probs2:", probs2)
    # print("grouping:", group_numbers(probs2, 0.2))

    # return ans
    # return ans2, probs2
    # print("ans2_final, probs2, probs2_final:", ans2_final, probs2, probs2_final, [CX, CY])

    return ans2_final, probs2_final, ans2


def kernel_determiner_bigger(I, CX, CY, UB, LB):
    KERNEL2 = [[[], [], [], [], [], [], [], [], [], [], [], [], []],
               [[], [], [], [], [], [], [], [], [], [], [], [], []],
               [[], [], [], [], [], [], [], [], [], [], [], [], []],
               [[], [], [], [], [], [], [], [], [], [], [], [], []],
               [[], [], [], [], [], [], [], [], [], [], [], [], []],
               [[], [], [], [], [], [], [], [], [], [], [], [], []],
               [[], [], [], [], [], [], [], [], [], [], [], [], []],
               [[], [], [], [], [], [], [], [], [], [], [], [], []],
               [[], [], [], [], [], [], [], [], [], [], [], [], []],
               [[], [], [], [], [], [], [], [], [], [], [], [], []],
               [[], [], [], [], [], [], [], [], [], [], [], [], []],
               [[], [], [], [], [], [], [], [], [], [], [], [], []],
               [[], [], [], [], [], [], [], [], [], [], [], [], []]]

    for y in range(13):
        for x in range(13):
            KERNEL2[y][x] = I[CY - 6 + y][CX - 6 + x]

    # print("KERNEL2 from bigger:", KERNEL2)

    dirs_or2 = {
        "N": [[0, 3], [0, 4], [0, 5], [0, 6], [0, 7], [0, 8], [0, 9], [1, 4], [1, 5], [1, 6], [1, 7], [1, 8], [2, 4],
              [2, 5], [2, 6], [2, 7], [2, 8], [3, 5], [3, 6], [3, 7], [4, 5], [4, 6], [4, 7], [5, 6]],
        "NE": [[0, 9], [0, 10], [0, 11], [0, 12], [1, 12], [2, 12], [3, 12], [1, 8], [1, 9], [1, 10], [1, 11], [2, 11],
               [3, 11], [4, 11], [2, 8], [2, 9], [2, 10], [3, 10], [4, 10], [3, 7], [3, 8], [3, 9], [4, 9], [5, 9],
               [4, 7], [4, 8], [5, 8], [5, 7]],
        "E": [[6, 7], [5, 8], [6, 8], [7, 8], [5, 9], [6, 9], [7, 9], [4, 10], [5, 10], [6, 10], [7, 10], [8, 10],
              [4, 11], [5, 11], [6, 11], [7, 11], [8, 11], [3, 12], [4, 12], [5, 12], [6, 12], [7, 12], [8, 12],
              [9, 12]],
        "SE": [[7, 7], [8, 7], [8, 8], [7, 8], [9, 7], [9, 8], [9, 9], [8, 9], [7, 9], [10, 8], [10, 9], [10, 10],
               [9, 10], [8, 10], [11, 8], [11, 9], [11, 10], [11, 11], [10, 11], [9, 11], [8, 11], [12, 9], [12, 10],
               [12, 11], [12, 12], [11, 12], [10, 12], [9, 12]],
        "S": [[7, 6], [8, 5], [8, 6], [8, 7], [9, 5], [9, 6], [9, 7], [10, 4], [10, 5], [10, 6], [10, 7], [10, 8],
              [11, 4], [11, 5], [11, 6], [11, 7], [11, 8], [12, 3], [12, 4], [12, 5], [12, 6], [12, 7], [12, 8],
              [12, 9]],
        "SW": [[7, 5], [7, 4], [8, 4], [8, 5], [7, 3], [8, 3], [9, 3], [9, 4], [9, 5], [8, 2], [9, 2], [10, 2], [10, 3],
               [10, 4], [8, 1], [9, 1], [10, 1], [11, 1], [11, 2], [11, 3], [11, 4], [9, 0], [10, 0], [11, 0], [12, 0],
               [12, 1], [12, 2], [12, 3]],
        "W": [[6, 5], [5, 4], [6, 4], [7, 4], [5, 3], [6, 3], [7, 3], [4, 2], [5, 2], [6, 2], [7, 2], [8, 2], [4, 1],
              [5, 1], [6, 1], [7, 1], [8, 1], [3, 0], [4, 0], [5, 0], [6, 0], [7, 0], [8, 0], [9, 0]],
        "NW": [[5, 5], [5, 4], [4, 4], [4, 5], [5, 3], [4, 3], [3, 3], [3, 4], [3, 5], [4, 2], [3, 2], [2, 2], [2, 3],
               [2, 4], [4, 1], [3, 1], [2, 1], [1, 1], [1, 2], [1, 3], [1, 4], [3, 0], [2, 0], [1, 0], [0, 0], [0, 1],
               [0, 2], [0, 3]]}

    probs2 = np.array(
        [np.average([KERNEL2[coords2[0]][coords2[1]] for coords2 in dirs_or2[d2]]) / 255 for d2 in dirs_or2.keys()])
    # print(probs2)

    # probs = np.array([np.sqrt(np.mean([KERNEL[coords[0]][coords[1]] ** 2 for coords in dirs_or[d]])) / 255 for d in dirs_or.keys()])
    # print(probs)

    # print(np.var(probs))
    # print(np.sqrt(np.mean(probs ** 2)))

    # ans2 = [dirs[index] for index in range(8) if probs[index] < THRESH]
    ans2 = [dirs[index2] for index2 in range(8) if (probs2[index2] < UB) and (probs2[index2] > LB)]

    # probs2_final = group_numbers(probs2, 0.2)[-1]
    probs2_final = probs2.tolist()

    ans2_final = [dirs[index] for index in range(8) if probs2[index] in probs2_final]

    # TODO: To later uncomment for debugging # print("ans2, probs2 from kernel_determiner_bigger:", [CX, CY], ans2, probs2)

    # print("probs2:", probs2)
    # print("grouping:", group_numbers(probs2, 0.2))

    # return ans
    # return ans2, probs2
    return ans2_final, probs2_final


def find_pyramidal_conical_structures(X, Y, Z, threshold=0.5):
    """
    Find pyramidal or conical structures in a 3D plot.

    Args:
        X, Y, Z (numpy arrays): The X, Y, and Z coordinates of the points in the 3D plot.
        threshold (float, optional): The threshold value to determine the height of the structure. Default is 0.5.

    Returns:
        List of 3D polygon collections representing the pyramidal or conical structures.
    """
    structures = []
    coords = []

    count = 0

    for i in range(X.shape[0] - 1):
        for j in range(X.shape[1] - 1):
            if Z[i, j] > threshold and Z[i + 1, j] > threshold and Z[i, j + 1] > threshold and Z[
                i + 1, j + 1] > threshold:
                vertices = np.array([(X[i, j], Y[i, j], Z[i, j]),
                                     (X[i + 1, j], Y[i + 1, j], Z[i + 1, j]),
                                     (X[i + 2, j], Y[i + 2, j], Z[i + 2, j]),
                                     (X[i + 1, j + 1], Y[i + 1, j + 1], Z[i + 1, j + 1]),
                                     (X[i + 2, j + 1], Y[i + 2, j + 1], Z[i + 2, j + 1]),
                                     (X[i + 2, j + 2], Y[i + 2, j + 2], Z[i + 2, j + 2]),
                                     (X[i + 1, j + 2], Y[i + 1, j + 2], Z[i + 1, j + 2]),
                                     (X[i, j + 1], Y[i, j + 1], Z[i, j + 1]),
                                     (X[i, j + 2], Y[i, j + 2], Z[i, j + 2])])
                poly = Poly3DCollection([vertices], alpha=0.5, edgecolor='black')
                structures.append(poly)

                # plt.figure(count)
                # print("count:", count, [j, i])
                # plt.imshow(vertices)

                # plt.show()
                # count += 1

                coords.append(vertices)

    return structures, coords


def slope(CX, CY, DIRECTION, IM):
    point_y, point_x = kernel_operator(DIRECTION, CY, CX, 1)

    delta_x = point_x - CX
    delta_y = point_y - CY

    # IM = IM.astype(float)

    # delta_z = (IM[point_y][point_x] - IM[CY][CX]) / 255
    # delta_z = IM[point_y][point_x] - IM[CY][CX]

    # distance = np.sqrt(delta_x**2 + delta_y**2 + delta_z**2)
    distance = np.sqrt(delta_x ** 2 + delta_y ** 2)

    # print("----------")
    # # print("delta_z, distance:", delta_z, distance, [CX, CY])
    # print("intensities:", (IM[CY][CX] - IM[point_y][point_x]) / distance)
    # print("----------")

    return (IM[CY][CX].astype(float) - IM[point_y][point_x].astype(float)) / distance

    # return (delta_x / distance, delta_y / distance, delta_z / distance)
    # return delta_z / distance


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
    # CX, CY = 332, 253

    # CX, CY = 320, 504

    data_2d = np.zeros([SIZE, SIZE])
    values = []

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

            # if temp_inner > max_value:
            #     max_value = temp_inner
            #     center_x, center_y = x, y

            if temp_inner > 127:
                values.append(temp_inner)

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

            if temp_inner > 127:
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
    # print("time for grapher_3D:", end_grapher - start_grapher)

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
            if True or ((curr[0] != X) and (i[0] != X) and (curr[1] != Y) and (i[1] != Y)):
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
                        ((angle_new_format < -0.125) and (angle_new_format > 0)):
                    direction_pointing = 'E'

                # print('direction_pointing before:', direction_pointing)

                if direction_pointing == 'SE':
                    if (curr[1] <= Y) or (i[1] <= Y):
                        direction_pointing = 'NW'
                elif direction_pointing == 'S':
                    if (curr[1] <= Y) or (i[1] <= Y):
                        direction_pointing = 'N'
                elif direction_pointing == 'SW':
                    if (curr[1] <= Y) or (i[1] <= Y):
                        direction_pointing = 'NE'
                elif direction_pointing == 'E':
                    if (curr[0] <= X) or (i[0] <= X):
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
        #                 ((angle_new_format < -0.125) and (angle_new_format > 0)):
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

        step = int(np.ceil(np.sqrt((max_coord[0] - COI_coords[0])**2 + (max_coord[1] - COI_coords[1])**2))) + 6
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
#                         ((angle_new_format < -0.125) and (angle_new_format > 0)):
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

def get_dirs_from_coords(_, __, ___, X, Y, PREV):
    # st_plot = time.time()
    cs = plt.contour(_, __, ___, colors='black')

    # img = np.zeros((100, 100), dtype=np.uint8)
    # img[25:75, 25:75] = 255
    # img[40:60, 40:60] = 0
    #
    # plt.figure("img")
    # plt.imshow(img)

    # find the contours of the array
    # contours, hierarchy = cv.findContours(np.array(___), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # print the number of contours found
    # print(f"Number of contours found: {len(contours)}")

    # contours, hierarchy = cv.findContours(___, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    # plt.figure("contours from opencv")
    # plt.imshow(contours)

    # print("contours form opencv:", contours)

    # end_plot = time.time()

    # print("time for plotting:", st_plot - end_plot)

    flag = 0
    index_path = 0

    coords = {}

    # print("something:", len(cs.collections))

    count_contours = 0

    while flag != 1:
        try:
            p = cs.collections[-3].get_paths()[index_path]
            plt.clf()
            v = p.vertices
            x = v[:, 0]
            y = v[:, 1]

            coords[index_path] = [[x[i], y[i]] for i in range(len(x))]

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

    for ii in range(len(coords)):
        c = coords[ii]

        # coords = COORDS[2]
        # coords = COORDS[1]
        # c = coords[0]

        # print("length of COORDS:", len(COORDS))

        # slopes = []
        degs = []
        degs_where = []
        direction = []

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
                        ((angle_new_format < -0.125) and (angle_new_format > 0)):
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

        # print("coords:", coords)
        # print("direction:", direction)

        temp = [item for item in direction if item]

        final_ans[ii] = list(np.unique(np.array(temp)))

    # print("final_ans:", final_ans)

    final_ans['loop'] = final_ans[0]

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

    # print("ans:", ans)

    return ans, PREV, count_contours, angle_new_format


def recenter(curr_coords, Harris_COI_coords, angle_next, step_wanted, Harris_directions, directions_grid, prev):
    Harris_COI_X, Harris_COI_Y = Harris_COI_coords
    curr_X, curr_Y = curr_coords
    angle_next = angle_next

    # print("Harris directions from recenter:", Harris_directions)

    # step_curr = max(abs(Harris_COI_X - curr_X), abs(Harris_COI_Y - curr_Y))
    step_curr = 5

    possible_curr = []
    possible_wanted = []

    for x in range(curr_X - step_wanted, curr_X + step_wanted + 1):
        for y in range(curr_Y - step_wanted, curr_Y + step_wanted + 1):
            possible_wanted.append([x, y])

    for x in range(curr_X - step_curr, curr_X + step_curr + 1):
        for y in range(curr_Y - step_curr, curr_Y + step_curr + 1):
            possible_curr.append([x, y])

    possible_cur_values = []
    # possible = []

    _, __, ___, X1, Y1 = grapher_3D(13, image_final, curr_coords[0], curr_coords[1])
    directions, pp, len_conts, angle = get_dirs_from_coords(_, __, ___, X1, Y1, None)

    # print("directions from recenter:", directions)

    X_, Y_ = X1, Y1

    flag_found = 0

    for position in possible_curr:
        if flag_found == 0:
    # for position in possible_wanted:
            delta_x = curr_coords[0] - position[0]
            delta_y = curr_coords[1] - position[1]
            theta_radians = np.arctan2(delta_y, delta_x)

            angle_new_format = theta_radians / np.pi
            perp1 = angle_next - 0.5
            perp2 = angle_next + 0.5
            # perp2 = 0.5 - angle_next

            # possible_cur_values.append(abs(angle_new_format - angle_next))

            # print("angle_new_format, angle_next:", angle_new_format, angle_next)

            if (angle_new_format == perp1) or (angle_new_format == perp2):
                # dirrs, prob_arr, n = kernel_determiner(image_final, position[0], position[1], 0.46, 0.3581)
                dirrs, prob_arr, __ = kernel_determiner(image_final, position[0], position[1], 1.1, 0.901)
                # print("dirrs dirrs:", dirrs, position)
                # print("dirrs __:", __, position)

                if __ == directions:
                # if (Harris_directions[0] in __) and (opp[prev] in __):
                    X_, Y_ = position[0], position[1]

                    print("dirrs dirrs:", dirrs, position)
                    print("dirrs __:", __, position, directions)

                    print("---------------------------------------------------------")

                    flag_found = 1

                # if Harris_directions[0] in __:
                #     possible.append(position)

    # minimum = min(possible_cur_values)
    # possible = [possible_curr[i] for i in range(len(possible_curr)) if possible_cur_values[i] == minimum]

    # print("possible:", possible, Harris_directions, possible_curr)
    # print("---------------------------------------------------------")

    # print("from recenter:", [X_, Y_])

    # X_new, Y_new = curr_X, curr_Y
    #
    # return X_new, Y_new

    return X_, Y_


terminal_coords = {}
Harris_corners_directions = {}
Harris_corners_angles = {}

start_terminal = timer()

for key in coords_for_COI:
    # start_sub_terminal = timer()
    coord = coords_for_COI[key]

    _, __, ___, X1, Y1 = grapher_3D(13, image_final, coord[0], coord[1])
    directions, pp, len_conts, angle = get_dirs_from_coords(_, __, ___, X1, Y1, None)

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


def mainFunction(X, Y, PREV, MODE, DIRECTIONS):
    # def mainFunction(X, Y, DIRECTIONS_INPUT, PROB_ARR_INPUT):
    #     prev = None
    prev = PREV

    directions = DIRECTIONS

    if MODE == 1:
        n, prob_arr, directions = kernel_determiner(image_final, X, Y, 1.1, 0.901)
        # print("directions, prob_arr:", directions, prob_arr, [X, Y], n)

    # if prev and opp[prev] in directions:
    #     directions.remove(opp[prev])

    print("entry level data:", directions, [X, Y], prev)
    # else:
    #     directions = Harris_corners_directions[[X, Y]]

    # directions, prob_arr, n = kernel_determiner(image_final, X, Y, 0.46, 0.3581)
    #
    # print("directions, prob_arr:", directions, prob_arr, [X, Y], n)

    # directions_T, prob_arr_T, n = kernel_determiner(image_final, X, Y, 1.1, -0.1)
    #
    # print("directions_T, prob_arr_T:", directions_T, prob_arr_T, [X, Y])

    # DIRECTIONS = None

    count = 0
    temp = None

    ans = ""

    while len(directions) != 0:

        if len(directions) == 1:
            # print("len(directions) = 1 at:", [X, Y], directions)
            if (not prev) or (directions[0] != opp[prev]):
                prev = directions[0]

                if not temp:
                    temp = directions[0]
                    count += 1
                elif temp == directions[0]:
                    if count < resolution:
                        count += 1
                    else:
                        # ans += "$" + str([X, Y]) + directions[0]
                        ans += "$" + directions[0]
                        count = 0
                        temp = None
                else:
                    count = 0
                    temp = directions[0]

                print("data at each point:", [X, Y], directions)

                Y, X = kernel_operator(directions[0], Y, X, 1)
                # directions, prob_arr, n = kernel_determiner(image_final, X, Y, 0.46, 0.3581)
                n, prob_arr, directions = kernel_determiner(image_final, X, Y, 1.1, 0.901)
            elif directions[0] == opp[prev]:
                _, __, ___, XX, YY = grapher_3D(13, image_final, X, Y)
                print("terminating now at:", [X, Y], [XX, YY], directions)

                key = None

                for k in terminal_coords:
                    if terminal_coords[k] == [XX, YY]:
                        key = k

                if not key:
                    for k in coords_for_COI:
                        if coords_for_COI[k] == [XX, YY]:
                            key = k

                    temp = Harris_corners_directions[key]
                    temp.remove(opp[prev])

                    directions = []
                    for item in temp:
                        pass
                    # directions.remove(opp[prev])
                else:
                    print("terminal point encountered at:", [X, Y])

            print("after data at each point, next point and directions:", [X, Y], directions)

        elif len(directions) > 1:
            if opp[prev] in directions:
                directions.remove(opp[prev])

            # print("directions after removing the opposite:", directions, prob_arr, [X, Y])

            directions_comparison = directions.copy()
            # directions_comparison is a copy of the directions array which contains only the next direction

            if len(directions) == 1:
                if not temp:
                    temp = directions[0]
                    count += 1
                elif temp == directions[0]:
                    if count < resolution:
                        count += 1
                    else:
                        # ans += "$" + str([X, Y]) + directions[0]
                        ans += "$" + directions[0]
                        count = 0
                        temp = None
                else:
                    count = 0
                    temp = directions[0]

                print("data at each point:", [X, Y], directions)

                Y, X = kernel_operator(directions[0], Y, X, 1)
                # directions, prob_arr, n = kernel_determiner(image_final, X, Y, 0.46, 0.3581)
                n, prob_arr, directions = kernel_determiner(image_final, X, Y, 1.1, 0.901)
            elif len(directions) > 1:
                _, __, ___, X1, Y1 = grapher_3D(13, image_final, X, Y)
                print("len(directions) > 1 at:", [X, Y], [X1, Y1], directions)

                # if ((X1, Y1) not in centers_traversing) or True:
                if (X1, Y1) not in centers_traversing:
                    # if (X1, Y1) not in centers_traversing:
                    #     centers_traversing[(X1, Y1)] = '-'

                    centers_traversing[(X1, Y1)] = '-'

                    directions, angle = None, None

                    for k in coords_for_COI:
                        tk = None
                        if coords_for_COI[k] == [X1, Y1]:
                            tk = k

                        if tk:
                            if tk in Harris_corners_directions:
                                directions = Harris_corners_directions[tk]
                                angle = Harris_corners_angles[tk]
                        else:
                            directions, pp, ____, angle = get_dirs_from_coords(_, __, ___, X1, Y1, prev)
                    # This directions is after removing the previous direction

                    print("prev at each point:", [X, Y], prev)

                    if opp[prev] in directions:
                        directions.remove(opp[prev])

                    if opp[prev] in directions_comparison:
                        directions_comparison.remove(opp[prev])

                    print("new directions at len(directions) > 1 at:", [X, Y], [X1, Y1], directions)

                    if len(directions) == 1:
                        # recenter function comes here
                        if not temp:
                            temp = directions[0]
                            count += 1
                        elif temp == directions[0]:
                            if count < resolution:
                                count += 1
                            else:
                                # ans += "$" + str([X, Y]) + directions[0]
                                ans += "$" + directions[0]
                                count = 0
                                temp = None
                        else:
                            count = 0
                            temp = directions[0]

                        print("data at each point:", [X, Y], directions)

                        X, Y = recenter([X, Y], [X1, Y1], angle, 1, directions, directions_comparison, prev)
                        n, prob_arr, directions = kernel_determiner(image_final, X, Y, 1.1, 0.901)
                        # pass
                    else:
                        # loop for multiple directions comes here
                        print("printed multiple directions at:", [X, Y], [X1, Y1], directions)
                        ans += "$$"

                        for i in range(len(directions)):
                            try:
                                Y_Temp, X_Temp = kernel_operator(directions[i], Y1, X1, getstep(directions[i], X1, Y1, dict_for_coords))

                                print("some data:", [X_Temp, Y_Temp], [X1, Y1], directions)
                                # print("some data:", [X1, Y1], [X1, Y1], directions)
                                mainFunction(X_Temp, Y_Temp, directions[i], 2, directions[i])
                                # mainFunction(X1, Y1, directions[i], 2, directions)
                            except:
                                print("error in the for loop at:", [X, Y], directions[i])
                        # pass
                else:
                    print("something has happened at:", [X, Y], [X1, Y1], directions)
                    break

                # TODO: Just a marker

                # # if ((X1, Y1) not in centers_traversing) or True:
                # if (X1, Y1) not in centers_traversing:
                #     # if (X1, Y1) not in centers_traversing:
                #     #     centers_traversing[(X1, Y1)] = '-'
                #
                #     centers_traversing[(X1, Y1)] = '-'
                #
                #     if len(directions) > 1:
                #         # print("printed multiple directions at:", [X, Y], [X1, Y1], directions, pp)
                #         print("printed multiple directions at:", [X, Y], [X1, Y1], directions)
                #         ans += "$$"
                #         # print("break at 1477")
                #
                #         for i in range(len(directions)):
                #             # X_Temp, Y_Temp = kernel_operator(directions[0], Y1, X1, 6)
                #             # Y_Temp, X_Temp = kernel_operator(directions[0], Y1, X1, 9)
                #             # Y_Temp, X_Temp = kernel_operator(directions[i], Y1, X1, 9)
                #             Y_Temp, X_Temp = kernel_operator(directions[i], Y1, X1, getstep(directions[i], X1, Y1, dict_for_coords))
                #
                #             print("some data:", [X_Temp, Y_Temp], [X1, Y1], directions)
                #
                #             # mainFunction(X_Temp, Y_Temp, opp[directions[0]], None)
                #             # mainFunction(X_Temp, Y_Temp, directions[0], None)
                #             # mainFunction(X_Temp, Y_Temp, directions[0], None)
                #             mainFunction(X_Temp, Y_Temp, directions[i], None)
                #         break
                #     else: # 'else' is for the case where there is only one direction more to go
                #         print("directions:", directions, directions_comparison, [X, Y])
                #         X, Y = recenter([X, Y], [X1, Y1], angle, 1, directions, directions_comparison, pp)
                #
                #         n, prob_arr, directions = kernel_determiner(image_final, X, Y, 1.1, 0.91)
                #
                #         print("new directions from else block:", directions, n, [X, Y])
                #         # print("break at 1487")
                #
                #         # if not temp:
                #         #     temp = directions[0]
                #         #     count += 1
                #         # elif temp == directions[0]:
                #         #     if count < resolution:
                #         #         count += 1
                #         #     else:
                #         #         # ans += "$" + str([X, Y]) + directions[0]
                #         #         ans += "$" + directions[0]
                #         #         count = 0
                #         #         temp = None
                #         # else:
                #         #     count = 0
                #         #     temp = directions[0]
                #         #
                #         # Y, X = kernel_operator(directions[0], Y, X, 1)
                #         # # directions, prob_arr, n = kernel_determiner(image_final, X, Y, 0.46, 0.3581)
                #         # n, prob_arr, directions = kernel_determiner(image_final, X, Y, 1.1, 0.91)
                #         #
                #         # print("new directions from else block:", directions, n)
                #
                #         # break
                #
                # # TODO: Uncomment from here for the remaining else block
                # else:
                #     print("an error occurred in traversing the points at:", [X1, Y1], [X, Y], directions)
                #
                #     break
            else:
                print("len(directions) == 0 at:", [X, Y])
                break
    else:
        print("len(directions) = 0:", directions, [X, Y])

    print("ans:", ans, [X, Y])


key = None
_, __, ___, X1, Y1 = grapher_3D(13, image_final, X, Y)
for k in coords_for_COI:
    if coords_for_COI[k] == [X1, Y1]:
        key = k

X, Y = recenter([X, Y], [X1, Y1], Harris_corners_angles[key], 1, None, None, None)
mainFunction(X, Y, None, 1, None)

print(juncs)
print("centers:", centers)
print("centers_traversing:", centers_traversing)

plt.figure("image_final_Harris")
plt.imshow(image_final_Harris)

plt.figure("grayscaled")
plt.imshow(image_final)

end = timer()
print("time taken for program to run:", end - start)
print("time taken for terminal loop to run:", end_terminal - start_terminal)

plt.show()

cv.waitKey(0)
cv.destroyAllWindows()
