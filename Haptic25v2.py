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

traversed = {}

resolution = 1  # decides the resolution with which the hand movements will happen unless interrupted by direction change
intensity_threshold = 127
ans = ""

centers_traversing = {}

image = cv.imread('test image.png')

avgBlur_thinner = cv.blur(image, (5, 5))
image_final_thinner = cv.cvtColor(avgBlur_thinner, cv.COLOR_BGR2GRAY)
image_final_thinner = np.array(image_final_thinner)

avgBlur = cv.medianBlur(image, 5)

image_final_t = cv.cvtColor(avgBlur, cv.COLOR_BGR2GRAY)
image_final_test = np.bitwise_not(image_final_t)
image_final_test = np.float32(image_final_test)
image_final = np.array(image_final_t)

maintainer_matrix = [[255 for i in range(len(image_final[0]))] for ii in range(len(image_final))]


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

dst = cv.cornerHarris(image_final_test, 2, 13,
                      0.05)  # image, block_size, kernel_size, k (Harris detector free parameter)

dst = cv.dilate(dst, None)
image_Harris = image.copy()
image_Harris[dst > 0.01 * dst.max()] = [0, 0, 255]

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

def get_dirs_from_coords(_, __, ___, X, Y, PREV):
    # st_plot = time.time()
    cs = plt.contour(_, __, ___, colors='black')
    ans_f = []
    direction_temp_count = {}
    count_contours_dict = {}

    end_dir_coords = {}

    for ci in range(len(cs.collections)):

        flag = 0
        index_path = 0

        coords = {}

        # print("something:", len(cs.collections))

        count_contours = 0
        v = None

        while flag != 1:
            try:
                # p = cs.collections[-3].get_paths()[index_path]
                p = cs.collections[ci].get_paths()[index_path]
                # p = cs.collections[1].get_paths()[index_path]
                # print("len(p) [X, Y]:", len(p), [X, Y])

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
                            end_dir_coords[direction_pointing] = [
                                [[round(prev[0], 0), round(prev[1], 0)], delta_y, delta_x]]

                        else:
                            end_dir_coords[direction_pointing].append(
                                [[round(prev[0], 0), round(prev[1], 0)], delta_y, delta_x])

                    if i == len(coords):
                        if direction_pointing not in end_dir_coords:
                            end_dir_coords[direction_pointing] = [
                                [[round(curr[0], 0), round(curr[1], 0)], delta_y, delta_x]]

                        else:
                            end_dir_coords[direction_pointing].append(
                                [[round(curr[0], 0), round(curr[1], 0)], delta_y, delta_x])

                    # if i == 1:
                    #     if direction_pointing not in end_dir_coords:
                    #         # end_dir_coords[direction_pointing] = {ci: [[[round(prev[0], 0), round(prev[1], 0)], delta_y, delta_x, ii, "1"]]}
                    #         end_dir_coords[direction_pointing] = {ci: [[[round(prev[0], 0), round(prev[1], 0)], delta_y, delta_x]]}
                    #
                    #     else:
                    #         if ci in end_dir_coords[direction_pointing]:
                    #             # end_dir_coords[direction_pointing][ci].append([[round(prev[0], 0), round(prev[1], 0)], delta_y, delta_x, ii, "1"])
                    #             end_dir_coords[direction_pointing][ci].append([[round(prev[0], 0), round(prev[1], 0)], delta_y, delta_x])
                    #
                    #         else:
                    #             # end_dir_coords[direction_pointing][ci] = [[[round(prev[0], 0), round(prev[1], 0)], delta_y, delta_x, ii, "1"]]
                    #             end_dir_coords[direction_pointing][ci] = [[[round(prev[0], 0), round(prev[1], 0)], delta_y, delta_x]]
                    #
                    # if i == len(coords):
                    #     if direction_pointing not in end_dir_coords:
                    #         # end_dir_coords[direction_pointing] = {ci: [[[round(curr[0], 0), round(curr[1], 0)], delta_y, delta_x, ii, "l"]]}
                    #         end_dir_coords[direction_pointing] = {ci: [[[round(curr[0], 0), round(curr[1], 0)], delta_y, delta_x]]}
                    #
                    #     else:
                    #         if ci in end_dir_coords[direction_pointing]:
                    #             # end_dir_coords[direction_pointing][ci].append([[round(curr[0], 0), round(curr[1], 0)], delta_y, delta_x, ii, "l"])
                    #             end_dir_coords[direction_pointing][ci].append([[round(curr[0], 0), round(curr[1], 0)], delta_y, delta_x])
                    #
                    #         else:
                    #             # end_dir_coords[direction_pointing][ci] = [[[round(curr[0], 0), round(curr[1], 0)], delta_y, delta_x, ii, "l"]]
                    #             end_dir_coords[direction_pointing][ci] = [[[round(curr[0], 0), round(curr[1], 0)], delta_y, delta_x]]

                    direction.append(direction_pointing)
                else:
                    direction.append(None)

            # print("coords:", coords, "inside get_dirs [X, Y]:", len(coords), [X, Y])
            # print("direction at:", [X, Y], direction, "inside get_dirs [X, Y]:", len(coords), [X, Y])

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

                        if vi == 1:
                            if direction_pointing not in end_dir_coords:
                                end_dir_coords[direction_pointing] = [
                                    [[round(prev[0], 0), round(prev[1], 0)], delta_y, delta_x]]

                            else:
                                end_dir_coords[direction_pointing].append(
                                    [[round(prev[0], 0), round(prev[1], 0)], delta_y, delta_x])

                        if vi == len(v):
                            if direction_pointing not in end_dir_coords:
                                end_dir_coords[direction_pointing] = [
                                    [[round(curr[0], 0), round(curr[1], 0)], delta_y, delta_x]]

                            else:
                                end_dir_coords[direction_pointing].append(
                                    [[round(curr[0], 0), round(curr[1], 0)], delta_y, delta_x])

                        # if vi == 1:
                        #     if direction_pointing not in end_dir_coords:
                        #         end_dir_coords[direction_pointing] = {
                        #             # ci: [[[round(prev[0], 0), round(prev[1], 0)], delta_y, delta_x, vi, "v1"]]}
                        #             ci: [[[round(prev[0], 0), round(prev[1], 0)], delta_y, delta_x]]}
                        #
                        #     else:
                        #         if ci in end_dir_coords[direction_pointing]:
                        #             end_dir_coords[direction_pointing][ci].append(
                        #                 # [[round(prev[0], 0), round(prev[1], 0)], delta_y, delta_x, vi, "v1"])
                        #                 [[round(prev[0], 0), round(prev[1], 0)], delta_y, delta_x])
                        #
                        #         else:
                        #             end_dir_coords[direction_pointing][ci] = [
                        #                 # [[round(prev[0], 0), round(prev[1], 0)], delta_y, delta_x, vi, "v1"]]
                        #                 [[round(prev[0], 0), round(prev[1], 0)], delta_y, delta_x]]
                        #
                        # if vi == len(coords):
                        #     if direction_pointing not in end_dir_coords:
                        #         end_dir_coords[direction_pointing] = {
                        #             # ci: [[[round(curr[0], 0), round(curr[1], 0)], delta_y, delta_x, vi, "vl"]]}
                        #             ci: [[[round(curr[0], 0), round(curr[1], 0)], delta_y, delta_x]]}
                        #
                        #     else:
                        #         if ci in end_dir_coords[direction_pointing]:
                        #             end_dir_coords[direction_pointing][ci].append(
                        #                 # [[round(curr[0], 0), round(curr[1], 0)], delta_y, delta_x, vi, "vl"])
                        #                 [[round(curr[0], 0), round(curr[1], 0)], delta_y, delta_x])
                        #
                        #         else:
                        #             end_dir_coords[direction_pointing][ci] = [
                        #                 # [[round(curr[0], 0), round(curr[1], 0)], delta_y, delta_x, vi, "vl"]]
                        #                 [[round(curr[0], 0), round(curr[1], 0)], delta_y, delta_x]]

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

    # ^^^^^^^^^^^^^^^^^^^^^^^^end_dir_coords operations^^^^^^^^^^^^^^^^^^^^^^^^

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
            if (image_final[int(coord[1])][int(coord[0])] < 100) and (np.linalg.norm(np.array([X, Y]) - np.array(coord))
                                                                      > np.linalg.norm(
                        np.array([X, Y]) - np.array(coord_max))):
                coord_max, dy_max, dx_max = coord, dy, dx

        end_dir_coords_tt[dir_key] = [[int(coord_max[0]), int(coord_max[1])], dy_max, dx_max]

    end_dir_coords = end_dir_coords_tt

    # print("end_dir_coords in get_dirs_from_coords:", [X, Y], end_dir_coords)

    end_dir_coords_tt_intersection = {}

    for dir_key in end_dir_coords:
        # pass
        # end_dir_coords_tt_intersection[dir_key] = POI(end_dir_coords[dir_key], [X, Y])

        # dy, dx = end_dir_coords[dir_key][1], end_dir_coords[dir_key][2]
        #
        # if dy == 0.0:
        #     dy = 10e-5
        #
        # if dx == 0.0:
        #     m1 = 10e-5
        #
        # else:
        #     m1 = ((-1) * dx) / dy

        pt = find_max_perp(
            end_dir_coords[dir_key][0], ((end_dir_coords[dir_key][1] / end_dir_coords[dir_key][2])
                                         if end_dir_coords[dir_key][2] != 0.0 else ((-1) * 10e5)), image_final, 100)

        print("data before specific:", end_dir_coords, pt, dir_key)

        end_dir_coords_tt_intersection[dir_key] = dirLR_β(image_final, pt[0], pt[1], 27, 100,
                                                          dir_key, PREV_COORDS=[X, Y], IGNORE_HARRIS=True)[-1][0]

        print("end_dir_coords in get_dirs_from_coords after POI operation specific:", [X, Y], pt, dir_key,
              end_dir_coords_tt_intersection[dir_key])

    end_dir_coords = end_dir_coords_tt_intersection

    print("end_dir_coords in get_dirs_from_coords after POI operation:", [X, Y], end_dir_coords)

    # ^^^^^^^^^^^^^^^^^^^^^^^^end_dir_coords operations^^^^^^^^^^^^^^^^^^^^^^^^

    return ans, PREV, max_count_contours, None, end_dir_coords

def find_max_perp(point, beta_bar, img_f, limit, length=13):
    # _, __, ___, X1, Y1 = grapher_3D(13, img_f, point[0], point[1])
    # return [X1, Y1]

    start_fmp = timer()

    possible = []

    xc, yc = point

    l_xc, u_xc = xc, xc
    l_yc, u_yc = yc, yc
    l_yc_d, u_yc_d = yc, yc

    range_of_points = []

    while (img_f[yc][l_xc - 1] < limit) and (abs(xc - l_xc + 2) <= length):
        l_xc -= 1

    while (img_f[yc][u_xc + 1] < limit) and (abs(u_xc - xc + 2) <= length):
        u_xc += 1

    while (img_f[l_yc - 1][l_xc] < limit) and (abs(yc - l_yc + 2) <= length):
        l_yc -= 1

    while (img_f[u_yc + 1][l_xc] < limit) and (abs(u_yc - yc + 2) <= length):
        u_yc += 1

    while (img_f[l_yc_d - 1][u_xc] < limit) and (abs(yc - l_yc_d + 2) <= length):
        l_yc_d -= 1

    while (img_f[u_yc_d + 1][u_xc] < limit) and (abs(u_yc_d - yc + 2) <= length):
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

    elif beta_bar == 0.0:
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

    print("data from find_max_perp at:", point, "thickness:", thickness, "final point:", median_from_possible,
          "beta_perp:", beta_perp)

    # ------------------------------------------------------------------------------------------------------------------

    end_fmp = timer()
    print("time taken for fmp:", end_fmp - start_fmp)

    return median_from_possible

def dirLR_β(IMG_F, CX, CY, KER_SIZE, LIMIT, PREV, PREV_COORDS=None, IGNORE_HARRIS=False):
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

    if IGNORE_HARRIS == False:
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

    if IGNORE_HARRIS == False:
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

    # print("final_QS in dirLR:", final_QS, PREV)

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

    print("beta_bar related data in dirLR_ß:", [CX, CY], beta_bar_fQS, beta_bar_num_fQS, beta_bar_denom_fQS, xmean_fQS,
          ymean_fQS, angle_fQS)
    # print("beta_bar_d related data in dirLR_ß:", [CX, CY], beta_bar_fQS_d, beta_bar_num_fQS, beta_bar_denom_fQS_d, xmean_fQS, angle_fQS_d)

    points_possible = []
    points_possible_fQS = []

    # print("values_coord in dirLR:", [CX, CY], values_coord)
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
                    points_possible_fQS.append([x, y])

                lower_y = np.floor(((x - CX) * beta_bar_fQS) + CY)
                upper_y = lower_y + 1

                if (y == upper_y) or (y == lower_y):
                    points_possible.append([x, y])
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

    print("PREV_COORDS data from dirLR_ß at:", [CX, CY], PREV_COORDS, final_direction_fQS, final_point_fQS)

    if PREV_COORDS:
        for i in range(len(final_direction_fQS)):
            coord = final_point_fQS[i]

            if np.linalg.norm(np.array(coord) - np.array(PREV_COORDS)) > \
                    np.linalg.norm(np.array([CX, CY]) - np.array(PREV_COORDS)):
                final_direction_fQS_tc.append(final_direction_fQS[i])
                final_point_fQS_tc.append(coord)

        if len(final_point_fQS_tc) != 0:
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
    directions, pp, len_conts, angle, _ = get_dirs_from_coords(_, __, ___, X1, Y1, None)

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

ans_final = []

plt.figure("image_final")
plt.imshow(image_final)


def mainFunction2(X, Y, PREV, DIRECTIONS, POINT_NEXT=None, prev_coords=None):
    if PREV and opp[PREV] in DIRECTIONS:
        DIRECTIONS.remove(opp[PREV])

    print(Fore.GREEN + "ans_ans_ans:", [X, Y], PREV, DIRECTIONS, Style.RESET_ALL)

    # temp_str = str(X) + ", " + str(Y)

    # ANS[temp_str] = DIRECTIONS

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

        print("data len(DIRECTIONS) == 1 at:", [X, Y], directions_testing, points_testing, PREV,
              opp[PREV] if PREV is not None else "None")

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
                    mainFunction2(points_testing[0][0], points_testing[0][1], DIRECTIONS[0], [directions_testing[0]],
                                  prev_coords=[X, Y])

                else:
                    print("len(directions_testing) == 0 some error occurred in dirLR at:", [X, Y], points_testing)
                    # print("relocated point at:", [X, Y], "is:", find_max_perp([X, Y], -0.9664187145904831, image_final, 100))

                    X_Reloc, Y_Reloc = find_max_perp([X, Y], beta_bar_testing, image_final, 100)
                    print("relocated point at:", [X, Y], "is:", [X_Reloc, Y_Reloc])

                    beta_bar_testing, angle_testing, directions_testing, points_testing = \
                        dirLR_β(image_final, X_Reloc, Y_Reloc, 27, 100, PREV, PREV_COORDS=prev_coords)

                    mainFunction2(points_testing[0][0], points_testing[0][1], DIRECTIONS[0], [directions_testing[0]],
                                  prev_coords=[X, Y])

            else:
                _, __, ___, X1, Y1 = grapher_3D(13, image_final, points_testing[0], points_testing[1])
                directions, pp, len_conts, angle, coords_for_next_dirs = get_dirs_from_coords(_, __, ___, X1, Y1, None)

                if PREV and (opp[PREV] in directions):
                    directions.remove(opp[PREV])

                print("multiple directions here at:", points_testing, directions)

                # for i in range(len(directions)):
                for key_nxt_coord in coords_for_next_dirs:
                    try:
                        # Y_Temp, X_Temp = kernel_operator(directions[i], Y1, X1,
                        #                                  getstep(directions[i], X1, Y1, dict_for_coords))

                        X_Temp, Y_Temp = coords_for_next_dirs[key_nxt_coord]

                        # print("data before applying dirLR_β at:", [X_Temp, Y_Temp], [X1, Y1], directions[i], "at harris:", points_testing)
                        print("data before applying dirLR_β at:", [X_Temp, Y_Temp], [X1, Y1], key_nxt_coord,
                              "at harris:", points_testing)

                        # Y_Temp, X_Temp = kernel_operator(directions[i], points_testing[1], points_testing[0],
                        #                                  getstep(directions[i], points_testing[1], points_testing[0], dict_for_coords))
                        #
                        # print("data before applying dirLR_β at:", [X_Temp, Y_Temp], [X1, Y1], directions[i])

                        # beta_bar_testing, angle_testing, directions_testing, points_testing = \
                        #     dirLR(image_final, X_Temp, Y_Temp, 13, 100, directions[i])

                        # # ----------------------------------USED SUCCESSFULLY----------------------------------
                        # beta_bar_testing, angle_testing, directions_testing, points_testing = \
                        #     dirLR_β(image_final, X_Temp, Y_Temp, 27, 100, directions[i], PREV_COORDS=prev_coords)
                        #
                        # print("data inside for loop at:", [X_Temp, Y_Temp], [X, Y], directions_testing, points_testing, directions[i],
                        #       opp[directions[i]] if directions[i] is not None else "None")
                        # # ----------------------------------USED SUCCESSFULLY----------------------------------

                        # mainFunction2(X_Temp, Y_Temp, directions[i], [directions_testing[0]], prev_coords=[X, Y])
                        mainFunction2(X_Temp, Y_Temp, key_nxt_coord, [directions_testing[0]], prev_coords=[X, Y])

                    except Exception as e:
                        print("some error occurred at:", points_testing, [X1, Y1], e)
    else:
        pass


key = None
_, __, ___, X1, Y1 = grapher_3D(13, image_final, XO, YO)

for k in coords_for_COI:
    if coords_for_COI[k] == [X1, Y1]:
        key = k

_, __, ___, X1, Y1 = grapher_3D(13, image_final, XO, YO)
directions, pp, ____, angle, _____ = get_dirs_from_coords(_, __, ___, X1, Y1, None)

# beta_bar_testing, angle_testing, direction_testing, point_testing = dirLR(image_final, X1, Y1, 13, 100, None)
beta_bar_testing, angle_testing, direction_testing, point_testing = dirLR_β(image_final, X1, Y1, 13, 100, None)

print("before starting:", [XO, YO], [X1, Y1], directions, direction_testing, point_testing)

start2 = timer()
mainFunction2(X1, Y1, None, direction_testing, POINT_NEXT=point_testing, prev_coords=None)
end2 = timer()
print("time taken for mainFunction2:", end2 - start2)

print(juncs)
print("centers:", centers)
print("centers_traversing:", centers_traversing)

plt.figure("image_final_Harris")
plt.imshow(image_final_Harris)

plt.figure("grayscaled")
plt.imshow(image_final)

values = {}

end = timer()
print("time taken for program to run:", end - start)
print("time taken for terminal loop to run:", end_terminal - start_terminal)

count_z = 0

for y in range(len(maintainer_matrix)):
    for x in range(len(maintainer_matrix[0])):
        if type(maintainer_matrix[y][x]) == str:
            maintainer_matrix[y][x] = 0

        if maintainer_matrix[y][x] == 0:
            count_z += 1

print("count_z:", count_z)

plt.figure("maintainer_matrix")
plt.imshow(maintainer_matrix)

plt.show()

cv.waitKey(0)
cv.destroyAllWindows()
