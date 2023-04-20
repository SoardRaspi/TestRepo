import time

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.cm as cm
from flowpy import *
from scipy.interpolate import RegularGridInterpolator

from timeit import default_timer as timer

dir_dis = {"N": np.array([0, -1]), "NE": np.array([+1, -1]), "E": np.array([+1, 0]), "SE": np.array([+1, +1]),
           "S": np.array([0, +1]), "SW": np.array([-1, +1]), "W": np.array([-1, 0]), "NW": np.array([-1, -1])}
dict_bool = {0: "N", 1: "NE", 2: "E", 3: "SE", 4: "S", 5: "SW", 6: "W", 7: "NW"}
dirs = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
opp = {"N": "S", "NE": "SW", "E": "W", "SE": "NW", "S": "N", "SW": "NE", "W": "E", "NW": "SE"}
basic = ["N", "E", "S", "W"]
juncs = {}

resolution = 6  # decides the resolution with which the hand movements wil occur unless interrupted by direction change
ans = ""

# dirs_or = {"N": [4, 3], "NE": [5, 3], "E": [5, 4], "SE": [5, 5],
#            "S": [4, 5], "SW": [3, 5], "W": [3, 4], "NW": [3, 3]}

# angles = {"N": [-np.pi/8, np.pi/8], "NE": [np.pi/8, 3*np.pi/8], "E": [3*np.pi/8, 5*np.pi/8], "SE": [5*np.pi/8, 7*np.pi/8],
#           "S": [7*np.pi/8, 9*np.pi/8], "SW": [9*np.pi/8, 11*np.pi/8], "W": [11*np.pi/8, 13*np.pi/8], "NW": [13*np.pi/8, 15*np.pi/8]}

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
    KERNEL2 = np.array([[I[CY-5][CX-5], I[CY-5][CX-4], I[CY-5][CX-3], I[CY-5][CX-2], I[CY-5][CX-1], I[CY-5][CX], I[CY-5][CX+1], I[CY-5][CX+2], I[CY-5][CX+3], I[CY-5][CX+4], I[CY-5][CX+5]],
                        [I[CY-4][CX-5], I[CY-4][CX-4], I[CY-4][CX-3], I[CY-4][CX-2], I[CY-4][CX-1], I[CY-4][CX], I[CY-4][CX+1], I[CY-4][CX+2], I[CY-4][CX+3], I[CY-4][CX+4], I[CY-4][CX+5]],
                        [I[CY-3][CX-5], I[CY-3][CX-4], I[CY-3][CX-3], I[CY-3][CX-2], I[CY-3][CX-1], I[CY-3][CX], I[CY-3][CX+1], I[CY-3][CX+2], I[CY-3][CX+3], I[CY-3][CX+4], I[CY-3][CX+5]],
                        [I[CY-2][CX-5], I[CY-2][CX-4], I[CY-2][CX-3], I[CY-2][CX-2], I[CY-2][CX-1], I[CY-2][CX], I[CY-2][CX+1], I[CY-2][CX+2], I[CY-2][CX+3], I[CY-2][CX+4], I[CY-2][CX+5]],
                        [I[CY-1][CX-5], I[CY-1][CX-4], I[CY-1][CX-3], I[CY-1][CX-2], I[CY-1][CX-1], I[CY-1][CX], I[CY-1][CX+1], I[CY-1][CX+2], I[CY-1][CX+3], I[CY-1][CX+4], I[CY-1][CX+5]],
                        [I[CY][CX-5],   I[CY][CX-4],   I[CY][CX-3],   I[CY][CX-2],   I[CY][CX-1],   I[CY][CX],   I[CY][CX+1],   I[CY][CX+2],   I[CY][CX+3],   I[CY][CX+4],   I[CY][CX+5]],
                        [I[CY+1][CX-5], I[CY+1][CX-4], I[CY+1][CX-3], I[CY+1][CX-2], I[CY+1][CX-1], I[CY+1][CX], I[CY+1][CX+1], I[CY+1][CX+2], I[CY+1][CX+3], I[CY+1][CX+4], I[CY+1][CX+5]],
                        [I[CY+2][CX-5], I[CY+2][CX-4], I[CY+2][CX-3], I[CY+2][CX-2], I[CY+2][CX-1], I[CY+2][CX], I[CY+2][CX+1], I[CY+2][CX+2], I[CY+2][CX+3], I[CY+2][CX+4], I[CY+2][CX+5]],
                        [I[CY+3][CX-5], I[CY+3][CX-4], I[CY+3][CX-3], I[CY+3][CX-2], I[CY+3][CX-1], I[CY+3][CX], I[CY+3][CX+1], I[CY+3][CX+2], I[CY+3][CX+3], I[CY+3][CX+4], I[CY+3][CX+5]],
                        [I[CY+4][CX-5], I[CY+4][CX-4], I[CY+4][CX-3], I[CY+4][CX-2], I[CY+4][CX-1], I[CY+4][CX], I[CY+4][CX+1], I[CY+4][CX+2], I[CY+4][CX+3], I[CY+4][CX+4], I[CY+4][CX+5]],
                        [I[CY+5][CX-5], I[CY+5][CX-4], I[CY+5][CX-3], I[CY+5][CX-2], I[CY+5][CX-1], I[CY+5][CX], I[CY+5][CX+1], I[CY+5][CX+2], I[CY+5][CX+3], I[CY+5][CX+4], I[CY+5][CX+5]]])

    dirs_or2 = {"N": [[0, 3], [0, 4], [0, 5], [0, 6], [0, 7], [1, 4], [1, 5], [1, 6], [2, 4], [2, 5], [2, 6], [3, 4], [3, 5], [3, 6], [4, 5]],
                "NE": [[4, 6], [3, 7], [2, 7], [2, 8], [3, 8], [1, 7], [1, 8], [1, 9], [2, 9], [3, 9], [0, 8], [0, 9], [0, 10], [1, 10], [2, 10]],
                "E": [[5, 6], [4, 7], [5, 7], [6, 7], [4, 8], [5, 8], [6, 8], [4, 9], [5, 9], [6, 9], [3, 10], [4, 10], [5, 10], [6, 10], [7, 10]],
                "SE": [[6, 6], [7, 7], [8, 7], [8, 8], [7, 8], [9, 7], [9, 8], [9, 9], [8, 9], [7, 9], [10, 8], [10, 9], [10, 10], [9, 10], [8, 10]],
                "S": [[6, 5], [7, 4], [7, 5], [7, 6], [8, 4], [8, 5], [8, 6], [9, 4], [9, 5], [9, 6], [10, 3], [10, 4], [10, 5], [10, 6], [10, 7]],
                "SW": [[6, 4], [7, 3], [7, 2], [8, 3], [8, 3], [7, 1], [8, 1], [9, 1], [9, 2], [9, 3], [8, 0], [9, 0], [10, 0], [10, 1], [10, 2]],
                "W": [[5, 4], [4, 3], [5, 3], [6, 3], [4, 2], [5, 2], [6, 2], [4, 1], [5, 1], [6, 1], [3, 0], [4, 0], [5, 0], [6, 0], [7, 0]],
                "NW": [[4, 4], [3, 3], [3, 2], [2, 2], [2, 3], [3, 1], [2, 1], [1, 1], [1, 2], [1, 3], [2, 0], [1, 0], [0, 0], [0, 1], [0, 2]]}

    probs2 = np.array([np.average([KERNEL2[coords2[0]][coords2[1]] for coords2 in dirs_or2[d2]]) / 255 for d2 in dirs_or2.keys()])
    # print(probs2)

    # probs = np.array([np.sqrt(np.mean([KERNEL[coords[0]][coords[1]] ** 2 for coords in dirs_or[d]])) / 255 for d in dirs_or.keys()])
    # print(probs)

    # print(np.var(probs))
    # print(np.sqrt(np.mean(probs ** 2)))

    # ans2 = [dirs[index] for index in range(8) if probs[index] < THRESH]
    ans2 = [dirs[index2] for index2 in range(8) if (probs2[index2] < UB) and (probs2[index2] > LB)]

    probs2_final = group_numbers(probs2, 0.2)[-1]
    ans2_final = [dirs[index] for index in range(8) if probs2[index] in probs2_final]

    print("ans2, probs2 from kernel_determiner:", [CX, CY], ans2, probs2)

    # print("probs2:", probs2)
    # print("grouping:", group_numbers(probs2, 0.2))

    # return ans
    # return ans2, probs2
    return ans2_final, probs2_final

def kernel_determiner_bigger(I, CX, CY, UB, LB):
    # KERNEL2 = np.array([[[], [], [], [], [], [], [], [], [], [], [], [], []],
    #                     [[], [], [], [], [], [], [], [], [], [], [], [], []],
    #                     [[], [], [], [], [], [], [], [], [], [], [], [], []],
    #                     [[], [], [], [], [], [], [], [], [], [], [], [], []],
    #                     [[], [], [], [], [], [], [], [], [], [], [], [], []],
    #                     [[], [], [], [], [], [], [], [], [], [], [], [], []],
    #                     [[], [], [], [], [], [], [], [], [], [], [], [], []],
    #                     [[], [], [], [], [], [], [], [], [], [], [], [], []],
    #                     [[], [], [], [], [], [], [], [], [], [], [], [], []],
    #                     [[], [], [], [], [], [], [], [], [], [], [], [], []],
    #                     [[], [], [], [], [], [], [], [], [], [], [], [], []],
    #                     [[], [], [], [], [], [], [], [], [], [], [], [], []],
    #                     [[], [], [], [], [], [], [], [], [], [], [], [], []]])

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

    dirs_or2 = {"N": [[0, 3], [0, 4], [0, 5], [0, 6], [0, 7], [0, 8], [0, 9], [1, 4], [1, 5], [1, 6], [1, 7], [1, 8], [2, 4], [2, 5], [2, 6], [2, 7], [2, 8], [3, 5], [3, 6], [3, 7], [4, 5], [4, 6], [4, 7], [5, 6]],
                "NE": [[0, 9], [0, 10], [0, 11], [0, 12], [1, 12], [2, 12], [3, 12], [1, 8], [1, 9], [1, 10], [1, 11], [2, 11], [3, 11], [4, 11], [2, 8], [2, 9], [2, 10], [3, 10], [4, 10], [3, 7], [3, 8], [3, 9], [4, 9], [5, 9], [4, 7], [4, 8], [5, 8], [5, 7]],
                "E": [[6, 7], [5, 8], [6, 8], [7, 8], [5, 9], [6, 9], [7, 9], [4, 10], [5, 10], [6, 10], [7, 10], [8, 10], [4, 11], [5, 11], [6, 11], [7, 11], [8, 11], [3, 12], [4, 12], [5, 12], [6, 12], [7, 12], [8, 12], [9, 12]],
                "SE": [[7, 7], [8, 7], [8, 8], [7, 8], [9, 7], [9, 8], [9, 9], [8, 9], [7, 9], [10, 8], [10, 9], [10, 10], [9, 10], [8, 10], [11, 8], [11, 9], [11, 10], [11, 11], [10, 11], [9, 11], [8, 11], [12, 9], [12, 10], [12, 11], [12, 12], [11, 12], [10, 12], [9, 12]],
                "S": [[7, 6], [8, 5], [8, 6], [8, 7], [9, 5], [9, 6], [9, 7], [10, 4], [10, 5], [10, 6], [10, 7], [10, 8], [11, 4], [11, 5], [11, 6], [11, 7], [11, 8], [12, 3], [12, 4], [12, 5], [12, 6], [12, 7], [12, 8], [12, 9]],
                "SW": [[7, 5], [7, 4], [8, 4], [8, 5], [7, 3], [8, 3], [9, 3], [9, 4], [9, 5], [8, 2], [9, 2], [10, 2], [10, 3], [10, 4], [8, 1], [9, 1], [10, 1], [11, 1], [11, 2], [11, 3], [11, 4], [9, 0], [10, 0], [11, 0], [12, 0], [12, 1], [12, 2], [12, 3]],
                "W": [[6, 5], [5, 4], [6, 4], [7, 4], [5, 3], [6, 3], [7, 3], [4, 2], [5, 2], [6, 2], [7, 2], [8, 2], [4, 1], [5, 1], [6, 1], [7, 1], [8, 1], [3, 0], [4, 0], [5, 0], [6, 0], [7, 0], [8, 0], [9, 0]],
                "NW": [[5, 5], [5, 4], [4, 4], [4, 5], [5, 3], [4, 3], [3, 3], [3, 4], [3, 5], [4, 2], [3, 2], [2, 2], [2, 3], [2, 4], [4, 1], [3, 1], [2, 1], [1, 1], [1, 2], [1, 3], [1, 4], [3, 0], [2, 0], [1, 0], [0, 0], [0, 1], [0, 2], [0, 3]]}

    probs2 = np.array([np.average([KERNEL2[coords2[0]][coords2[1]] for coords2 in dirs_or2[d2]]) / 255 for d2 in dirs_or2.keys()])
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

    print("ans2, probs2 from kernel_determiner_bigger:", [CX, CY] ,ans2, probs2)

    # print("probs2:", probs2)
    # print("grouping:", group_numbers(probs2, 0.2))

    # return ans
    # return ans2, probs2
    return ans2_final, probs2_final

def retrace(IF, DIRS, JUNCS, JX, JY):
    JX_temp, JY_temp = 0, 0

    for branch_direction in DIRS:
        if branch_direction in basic:
            JY_temp, JX_temp = kernel_operator(branch_direction, JY, JX, 11)
        else:
            JY_temp, JX_temp = kernel_operator(branch_direction, JY, JX, 5)

        PREV = None
        DIRECTIONS, PROB_ARR = kernel_determiner(IF, JX_temp, JY_temp, 0.46, 0.3581)
        print("DIRECTIONS:", DIRECTIONS, branch_direction, [JX_temp, JY_temp])

        if not DIRECTIONS:
            print("PROB_ARR:", PROB_ARR)

        # while len(DIRECTIONS) != 0:
        #     print(DIRECTIONS)
        #
        #     if len(DIRECTIONS) == 1:
        #         PREV = DIRECTIONS[0]
        #
        #         JY_temp, JX_temp = kernel_operator(DIRECTIONS[0], JY_temp, JX_temp, 5)
        #         DIRECTIONS, PROB_ARR = kernel_determiner(IF, JX_temp, JY_temp, 0.46, 0.3581)
        #
        #         # if opp[prev] in directions:
        #         #     directions.remove(opp[prev])
        #         print([JX_temp, JY_temp])
        #     elif len(DIRECTIONS) > 1:
        #         if PREV and opp[PREV] in DIRECTIONS:
        #             DIRECTIONS.remove(opp[PREV])
        #
        #         # print(directions)
        #
        #         if len(DIRECTIONS) > 1:
        #             print("MULTIPLE AT:", [JX_temp, JY_temp])
        #             JUNCS[(X, Y)] = DIRECTIONS
        #             retrace(IF, DIRECTIONS, JUNCS, JX_temp, JY_temp)
        #
        #             break
        #     elif not DIRECTIONS:
        #         print("ERROR AT:", [JX_temp, JY_temp])
        #         print("PROB_ARR:", PROB_ARR)
        #         break
        #     else:
        #         print("MULTIPLE AT:", [JX_temp, JY_temp])
        #         print("PROB_ARR:", PROB_ARR)
        #         break

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
            if Z[i, j] > threshold and Z[i+1, j] > threshold and Z[i, j+1] > threshold and Z[i+1, j+1] > threshold:
                vertices = np.array([(X[i, j], Y[i, j], Z[i, j]),
                                     (X[i+1, j], Y[i+1, j], Z[i+1, j]),
                                     (X[i+2, j], Y[i+2, j], Z[i+2, j]),
                                     (X[i+1, j+1], Y[i+1, j+1], Z[i+1, j+1]),
                                     (X[i+2, j+1], Y[i+2, j+1], Z[i+2, j+1]),
                                     (X[i+2, j+2], Y[i+2, j+2], Z[i+2, j+2]),
                                     (X[i+1, j+2], Y[i+1, j+2], Z[i+1, j+2]),
                                     (X[i, j+1], Y[i, j+1], Z[i, j+1]),
                                     (X[i, j+2], Y[i, j+2], Z[i, j+2])])
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
    distance = np.sqrt(delta_x**2 + delta_y**2)

    # print("----------")
    # # print("delta_z, distance:", delta_z, distance, [CX, CY])
    # print("intensities:", (IM[CY][CX] - IM[point_y][point_x]) / distance)
    # print("----------")

    return (IM[CY][CX].astype(float) - IM[point_y][point_x].astype(float)) / distance

    # return (delta_x / distance, delta_y / distance, delta_z / distance)
    # return delta_z / distance

def find_least_inclined_line(direction_vectors):
    min_magnitude = float('inf')
    least_inclined_index = -1

    # Loop through each direction vector in the list
    for i, direction_vector in enumerate(direction_vectors):
        # Calculate the magnitude of the direction vector
        magnitude = np.sqrt(direction_vector[0] ** 2 + direction_vector[1] ** 2 + direction_vector[2] ** 2)

        # Compare the magnitude with the current min magnitude
        if magnitude < min_magnitude:
            min_magnitude = magnitude
            least_inclined_index = i

    return least_inclined_index


def find_most_inclined_line(direction_vectors):
    max_magnitude = -1
    most_inclined_index = -1

    # Loop through each direction vector in the list
    for i, direction_vector in enumerate(direction_vectors):
        # Calculate the magnitude of the direction vector
        magnitude = np.sqrt(direction_vector[0] ** 2 + direction_vector[1] ** 2 + direction_vector[2] ** 2)

        # Compare the magnitude with the current max magnitude
        if magnitude > max_magnitude:
            max_magnitude = magnitude
            most_inclined_index = i

    return most_inclined_index

def create_function_from_3d_plot(x, y, z, data):
    interpolator = RegularGridInterpolator((x, y, z), data)

    def func(x_val, y_val):
        return interpolator((x_val, y_val, 0))

    return func

def grapher_3D(SIZE, IMG_F, CX, CY, DIRECTIONS):
    data_2d = np.zeros([SIZE, SIZE])
    values = []
    dirs = {'N': [],
            'NE': [],
            'E': [],
            'SE': [],
            'S': [],
            'SW': [],
            'W': [],
            'NW': []}

    center_x, center_y = 0, 0
    max_value = 0

    # for y in range(max(len(image_final), len(image_final[0]))):
    #     for x in range(max(len(image_final), len(image_final[0]))):
    #         data_2d[y][x] = 0
    max_point = 0

    mid = (SIZE - 1) // 2

    for y in range(SIZE):
        for x in range(SIZE):
            temp_inner = 255 * np.sin((255 - IMG_F[CY - mid + y][CX - mid + x]) / 255)
            # data_2d[y][x] = np.sqrt((255 - IMG_F[CY - mid + y][CX - mid + x])**2 + (255 - IMG_F[CY - mid + y + 1][CX - mid + x])**2 + (255 - IMG_F[CY - mid + y - 1][CX - mid + x])**2)
            data_2d[y][x] = temp_inner

            if temp_inner > max_value:
                max_value = temp_inner
                center_x, center_y = x, y

            if temp_inner > 127:
                values.append(temp_inner)

            if temp_inner > max_point:
                max_point = temp_inner

    diff_x = center_x - 6
    diff_y = center_y - 6
    print("diffs:", diff_x, diff_y)

    values = [*set(values)]
    values.sort(reverse=True)
    print("values:", values)

    # center = [center_x, center_y]
    del values[0]

    for value in values:
        pass

    # Create X, Y coordinates using numpy meshgrid
    # X, Y = np.meshgrid(np.arange(data_2d.shape[0]), np.arange(data_2d.shape[1]))

    # X_grid, Y_grid = np.meshgrid(np.arange(CX - mid, CX + mid + 1, 1), np.arange(CY - mid, CY + mid + 1, 1))

    # CX += diff_x
    # CY += diff_y

    # for y in range(SIZE):
    #     for x in range(SIZE):
    #         temp_inner = 255 * np.sin((255 - IMG_F[CY - mid + y][CX - mid + x]) / 255)
    #         data_2d[y][x] = temp_inner

    CX += diff_x
    CY += diff_y

    for y in range(SIZE):
        for x in range(SIZE):
            temp_inner = 255 * np.sin((255 - IMG_F[CY - mid + y][CX - mid + x]) / 255)
            data_2d[y][x] = temp_inner

    X_grid, Y_grid = np.meshgrid(np.arange(CX - mid, CX + mid + 1, 1), np.arange(CY - mid, CY + mid + 1, 1))
    # X_grid, Y_grid = np.meshgrid(np.arange(CX - mid, CX + mid + 1, 1), np.arange(CY - mid, CY + mid + 1, 1))

    # X_grid, Y_grid = np.meshgrid(np.arange(CX - mid, CX + mid + 1, 1), np.arange(CY + mid + 1, CY - mid, -1))

    # X_Y_Spline = make_interp_spline(X, Y)

    # X = np.linspace(X.min(), X.max(), 500)
    # Y = X_Y_Spline(X)

    # Create a 3D figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the 2D array as a surface plot
    ax.plot_surface(X_grid, Y_grid, data_2d)
    ax.set_ylim(CY + mid + 1, CY - mid)

    # Set labels and title
    ax.set_xlabel('X Axis Label')
    ax.set_ylabel('Y Axis Label')
    ax.set_zlabel('Z Axis Label')
    ax.set_title('3D Plot centered at (' + str(CX) + ', ' + str(CY) + ')')
    # ax.set_title('3D Plot centered at (' + str(center_x) + ', ' + str(center_y) + ')')

    print("max value of the plot:", max_point)
    print("data_2D:", data_2d)

    print("new CX and CY:", [CX, CY])

    # U = np.random.randn(13, 13, 13)  # Example vector data for water flow in X direction
    # V = np.random.randn(13, 13, 13)  # Example vector data for water flow in Y direction
    # W = np.random.randn(13, 13, 13)  # Example vector data for water flow in Z direction
    # ax.quiver(X, Y, data_2d, U, V, W, length=0.1, normalize=True, color='r')

    # flow_direction = calculate_flow_direction(CX, CY, IMG_F[CY][CX], X_grid, Y_grid, data_2d)
    # flow_direction = calculate_flow_direction(CX, CY, IMG_F[CY][CX])
    # print("Flow direction at ({}, {}, {}): {}".format(CX, CY, IMG_F[CY][CX], flow_direction))

    # slopes = [slope(CX, CY, IMG_F[CY][CX], np.arange(CX - mid, CX + mid + 1, 1), np.arange(CY - mid, CY + mid + 1, 1),
    #                 data_2d, DIRECTION, IMG_F) for DIRECTION in DIRECTIONS]
    # slopes = []
    # return slopes

    # func = create_function_from_3d_plot(X_grid, Y_grid, IMG_F[Y_grid][X_grid], data_2d)
    # print("result:", func())

image = cv.imread('test image.png')

# W = 500
# H = int((852 / 861) * W)
# image = cv.resize(image, (W, H))

avgBlur = cv.blur(image, (5, 5))
# avgBlur2 = cv.blur(image, (3, 3))
# avgBlur = cv.blur(image, (3, 3))

# kernel = np.ones((3, 3), np.uint8)
# or_er = cv.erode(image, kernel, cv.BORDER_REFLECT)
# avg_er = cv.erode(avgBlur, kernel, iterations=1)

# avgBlur3 = cv.blur(avg_er, (2, 2))

# or_image = cv.bitwise_or(avgBlur, avgBlur2, mask=None)

# plt.figure("avgBlur")
# plt.imshow(avgBlur)

# plt.figure("or_er")
# plt.imshow(or_er)

# plt.figure("avg_er")
# plt.imshow(avg_er)

# plt.figure("avgBlur2")
# plt.imshow(avgBlur2)

# plt.figure("avgBlur3")
# plt.imshow(avgBlur3)

# plt.figure("or_image")
# plt.imshow(or_image)

# cv.imshow("original image", image)

############################################

# X, Y = 118, 76
X, Y = 201, 129
# X, Y = 289, 146

image_final = cv.cvtColor(avgBlur, cv.COLOR_BGR2GRAY)
image_final = np.array(image_final)

# print(kernel_determiner(image_final, 188, 146, 0.467))
# print(kernel_determiner(image_final, 117, 75, 0.467))
# print(kernel_determiner(image_final, 114, 72, 0.467))
# print(kernel_determiner(image_final, 118, 76, 0.467))
# print(kernel_determiner(image_final, 186, 292, 0.467))

# print(kernel_determiner(image_final, 118, 76, 0.467))
prev = None
directions, prob_arr = kernel_determiner(image_final, X, Y, 0.46, 0.3581)
print(directions, [X, Y])

# grapher_3D(13, image_final, 101, 724, None)

# grapher_3D(13, image_final, X, Y, None)
#
# print("slopes:", opp[dict_bool[find_least_inclined_line([slope(X, Y, DIRECTION, image_final) for DIRECTION in opp.keys()])]], opp[dict_bool[find_most_inclined_line([slope(X, Y, DIRECTION, image_final) for DIRECTION in opp.keys()])]])
#
# for i in range(10):
#     direction_next = opp[dict_bool[find_most_inclined_line([slope(X, Y, DIRECTION, image_final) for DIRECTION in opp.keys()])]]
#     Y, X = kernel_operator(direction_next, Y, X, 5)
#     print([X, Y])

# Y, X = kernel_operator(directions[0], Y, X, 4)
# directions = kernel_determiner(image_final, X, Y, 0.467)
#
# print("initial:", [X, Y])
# print(directions)

# grapher_3D(13, image_final, 322, 252, None)

count = 0
temp = None

while len(directions) != 0:
    print("entry level:", directions, [X, Y])

    # grapher_3D(13, image_final, X, Y, directions)

    if len(directions) == 1:
        prev = directions[0]

        if not temp:
            temp = directions[0]
            count += 1
        elif temp == directions[0]:
            if count < resolution:
                count += 1
            else:
                ans += "$" + str([X, Y]) + directions[0]
                count = 0
                temp = None
        else:
            count = 0
            temp = directions[0]

        # if directions[0] in basic:
        #     Y, X = kernel_operator(directions[0], Y, X, 5)
        # else:
        #     Y, X = kernel_operator(directions[0], Y, X, 11)

        Y, X = kernel_operator(directions[0], Y, X, 1)
        directions, prob_arr = kernel_determiner(image_final, X, Y, 0.46, 0.3581)
        # directions, prob_arr = kernel_determiner(image_final, X, Y, 1, 0)

        # if opp[prev] in directions:
        #     directions.remove(opp[prev])
        print([X, Y])
    elif len(directions) > 1:
        print("directions:", directions)
        if opp[prev] in directions:
            directions.remove(opp[prev])

        print("len(directions) > 1, [X, Y]:", directions, [X, Y])

        if len(directions) == 1:
            if not temp:
                temp = directions[0]
                count += 1
            elif temp == directions[0]:
                if count < resolution:
                    count += 1
                else:
                    ans += "$" + str([X, Y]) + directions[0]
                    count = 0
                    temp = None
            else:
                count = 0
                temp = directions[0]

            Y, X = kernel_operator(directions[0], Y, X, 1)
            directions, prob_arr = kernel_determiner(image_final, X, Y, 0.46, 0.3581)
            # directions, prob_arr = kernel_determiner(image_final, X, Y, 1, 0)
        else:
            print("multiple directions:", directions, [X, Y], prob_arr)

            directions_temp = []
            directions_temp = directions

            # for possible_dir in directions:
            #     possible_Y, possible_X = kernel_operator(possible_dir, Y, X, 1)
            #     # if image_final[possible_Y][possible_X] == image_final[Y][X]: #<-------------|
            #     if image_final[possible_Y][possible_X] <= image_final[Y][X]: #                |
            #         directions_temp.append(possible_dir) #                                    |
#                                                                                             |
            # if not directions_temp:                                           <-------------|
            #     for possible_dir in directions:                                             |
            #         possible_Y, possible_X = kernel_operator(possible_dir, Y, X, 1)         |
            #         if image_final[possible_Y][possible_X] <= image_final[Y][X]:            |
            #             directions_temp.append(possible_dir)                                |
#                                                                                             |
            # directions = directions_temp                                      <-------------|
            # del directions_temp

            if len(directions_temp) > 1:
                # Test 2
                if directions_temp == directions:
                    count_dict = {}

                    for possible_branch in directions_temp:
                        possible_branch_Y, possible_branch_X = kernel_operator(possible_branch, Y, X, 1)
                        possible_branch_direction, possible_branch_prob_arr = kernel_determiner_bigger(image_final, possible_branch_X, possible_branch_Y, 0.46, 0.3581)

                        if opp[possible_branch] in possible_branch_direction:
                            possible_branch_prob_arr.remove(possible_branch_prob_arr[possible_branch_direction.index(opp[possible_branch])])
                            possible_branch_direction.remove(opp[possible_branch])

                        print("branch, direction, prob_arr:", possible_branch, possible_branch_direction, possible_branch_prob_arr)

                        count_dict[possible_branch] = [possible_branch_direction[index] for index in range(len(possible_branch_direction)) if possible_branch_prob_arr[index] < 0.46]
                        count_dict[possible_branch] = len(count_dict[possible_branch])

                        print("count_dict:", count_dict)

                    counts = [count_dict[dirdir] for dirdir in count_dict.keys()]
                    DIRECTIONS = [dirdir for dirdir in count_dict.keys() if count_dict[dirdir] == max(counts)]

                    print("DIRECTIONS:", DIRECTIONS)

                    if len(DIRECTIONS) != 1:
                        print("directions from count_dict:", DIRECTIONS)

                        count = 0
                        ans += "$" + str([X, Y]) + temp + str(count) + "$M"
                        temp = None

                        # print("print at 551")
                        grapher_3D(13, image_final, X, Y, directions)
                        DIRECTIONS_TEMP = [direc for direc in DIRECTIONS if slope(X, Y, direc, image_final) > 0]
                        print("DIRECTIONS_TEMP:", DIRECTIONS_TEMP, [X, Y])
                        print("slopes:", [slope(X, Y, direc, image_final) for direc in DIRECTIONS])
                        # slope(X, Y, 'N', image_final)

                        directions = DIRECTIONS
                        print("direction(s) at:", [X, Y], DIRECTIONS)
                        Y, X = kernel_operator(directions[0], Y, X, 1)

                        DIRECTIONS, PROB_ARR = kernel_determiner_bigger(image_final, X, Y, 0.46, 0.3581)
                        print("DIRECTIONS, PROB_ARR right before the break:", DIRECTIONS, PROB_ARR, [X, Y])

                        break
                    else:
                        if not temp:
                            temp = DIRECTIONS[0]
                            count += 1
                        elif temp == DIRECTIONS[0]:
                            if count < resolution:
                                count += 1
                            else:
                                ans += "$" + str([X, Y]) + DIRECTIONS[0]
                                count = 0
                                temp = None
                        else:
                            count = 0
                            temp = DIRECTIONS[0]

                        directions = DIRECTIONS
                        print("direction(s) at:", [X, Y], DIRECTIONS)

                    # DIRECTIONS, PROB_ARR = kernel_determiner_bigger(image_final, X, Y, 0.46, 0.3581)
                    # if opp[prev] in DIRECTIONS:
                    #     DIRECTIONS.remove(opp[prev])
                    # print("DIRECTIONS, PROB_ARR:", DIRECTIONS, PROB_ARR)

                    # if len(DIRECTIONS) == 1:
                    #     directions = DIRECTIONS

                    # print(directions)

                    # Test 2
                    # print("time for Test 2")
                    # break
            elif len(directions_temp) == 1:
                if not temp:
                    temp = directions_temp[0]
                elif temp == prev:
                    if count < resolution:
                        count += 1
                    else:
                        ans += "$" + str([X, Y]) + directions_temp[0]
                        count = 0
                        temp = None
                else:
                    count = 0
                    temp = directions_temp[0]

                print("single direction:", directions_temp)
                directions = directions_temp
                print("direction(s) at:", [X, Y], DIRECTIONS)
                # break
            else:
                print("something else has happened:", directions_temp, directions)

                grapher_3D(13, image_final, X, Y, directions)

                # print("print at 578")
                break
            # break
    elif not directions:
        print("error at:", [X, Y])
        # print("prob_arr:", prob_arr)

        # print("print at 585")
        break

print(juncs)

print("--------------------")
print("and:", ans, [X, Y])
print("--------------------")

plt.figure("grayscaled")
plt.imshow(image_final)

plt.show()

cv.waitKey(0)
cv.destroyAllWindows()
