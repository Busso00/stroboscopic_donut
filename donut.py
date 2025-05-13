import numpy as np
import math
from time import sleep
import time, os
from threading import Semaphore

tick_time = []
lock = Semaphore(1)

DEBUG = False

RESOLUTION = 64  # suggested double of MAX_RESOLUTION

x = np.linspace(-1, 1, RESOLUTION, dtype=np.float32)
y = np.linspace(-1, 1, RESOLUTION, dtype=np.float32)
z = np.linspace(-1, 1, RESOLUTION, dtype=np.float32)

DEGREE = 3
PASS = int(360 / DEGREE)
LIGHTS = [".", ",", "-", "~", ":", ";", "=", "!", "*", "#", "$", "@"]
LIGHT_RANGE = len(LIGHTS) - 1
MAX_CHARS = 64

TOL = 2
EPS = TOL / RESOLUTION
WAIT_UNIFORM = 1 / 30


def gen_xyz():  # don't care for efficiency: just called once

    significative_poins = []
    normal_points = []

    R = 0.7  # center of the tube distance from (0,0)
    r1 = 0.34  # tube radius (external) form
    # r0 = r1 - 0.03/((RESOLUTION/TOL)/MAX_CHARS) #border width: OK, is around 0.03, all for low resolution
    r0 = 0

    Rs1 = r1 + 0.20
    Rs0 = Rs1 - 0.06 / (RESOLUTION / MAX_CHARS)  # border width: OK, is around 0.03

    xs = 0
    ys = R + r1

    r12 = r1**2
    r02 = r0**2

    Rs12 = Rs1**2
    Rs02 = Rs0**2

    R2 = R**2

    for zk in z:

        zk2 = zk**2
        if zk2 > r12:
            continue

        for xi in x:
            xi2 = xi**2
            xi_s2 = (xi - xs) ** 2

            for yj in y:
                yj2 = yj**2
                yj_s2 = (yj - ys) ** 2

                t1_toro = (R - math.sqrt(xi2 + yj2)) ** 2

                xiyj2 = xi_s2 + yj_s2

                if t1_toro + zk2 < r12:

                    if t1_toro + zk2 > r02:

                        if xiyj2 > Rs02:

                            r1_real = math.sqrt(t1_toro + zk2)

                            u = math.asin(zk / r1_real)  # parametric from
                            # u is angle z/(x,y)

                            if (xi2 + yj2) < R2:
                                u = -u + math.pi

                            const_sin = yj / (R + r1_real * math.cos(u))

                            v = math.asin(const_sin)  # parametric from
                            # v is angle x/y

                            if (xi / (R + r1_real * math.sin(u))) < 0:
                                v = -v + math.pi

                            grad_x = np.array(
                                [
                                    -r1_real * math.sin(u) * math.cos(v),
                                    -(R + r1_real * math.cos(u)) * math.sin(v),
                                ],
                                dtype=np.float32,
                            )
                            grad_y = np.array(
                                [
                                    -r1_real * math.sin(u) * math.sin(v),
                                    (R + r1_real * math.cos(u)) * math.cos(v),
                                ],
                                dtype=np.float32,
                            )
                            grad_z = np.array(
                                [r1_real * math.cos(u), 0], dtype=np.float32
                            )

                            N_M_x = np.linalg.det(
                                np.array(
                                    [[grad_y[0], grad_z[0]], [grad_y[1], grad_z[1]]],
                                    dtype=np.float32,
                                )
                            )

                            N_M_y = -np.linalg.det(
                                np.array(
                                    [[grad_x[0], grad_z[0]], [grad_x[1], grad_z[1]]],
                                    dtype=np.float32,
                                )
                            )

                            N_M_z = np.linalg.det(
                                np.array(
                                    [[grad_x[0], grad_y[0]], [grad_x[1], grad_y[1]]],
                                    dtype=np.float32,
                                )
                            )

                            norm = math.sqrt(
                                (
                                    np.array([N_M_x, N_M_y, N_M_z], dtype=np.float32)
                                    ** 2
                                ).sum()
                            )
                            norm_v = np.array(
                                [N_M_x / norm, N_M_y / norm, N_M_z / norm],
                                dtype=np.float32,
                            )
                            significative_poins.append([xi, yj, zk])
                            normal_points.append(norm_v)

                    if xiyj2 < Rs12:

                        if xiyj2 > Rs02:

                            if t1_toro + zk2 < r02:  # if internal

                                Rs = math.sqrt(xiyj2)

                                v = math.asin((xi - xs) / Rs)  # longitude

                                if yj - ys < 0:
                                    v = -v + math.pi

                                grad_x = np.array(
                                    [Rs * math.cos(v), 0], dtype=np.float32
                                )
                                grad_y = np.array(
                                    [-Rs * math.sin(v), 0], dtype=np.float32
                                )
                                grad_z = np.array([0, 1], dtype=np.float32)

                                N_M_x = np.linalg.det(
                                    np.array(
                                        [
                                            [grad_y[0], grad_z[0]],
                                            [grad_y[1], grad_z[1]],
                                        ],
                                        dtype=np.float32,
                                    )
                                )

                                N_M_y = -np.linalg.det(
                                    np.array(
                                        [
                                            [grad_x[0], grad_z[0]],
                                            [grad_x[1], grad_z[1]],
                                        ],
                                        dtype=np.float32,
                                    )
                                )

                                N_M_z = np.linalg.det(
                                    np.array(
                                        [
                                            [grad_x[0], grad_y[0]],
                                            [grad_x[1], grad_y[1]],
                                        ],
                                        dtype=np.float32,
                                    )
                                )

                                norm = math.sqrt(
                                    (
                                        np.array(
                                            [N_M_x, N_M_y, N_M_z], dtype=np.float32
                                        )
                                        ** 2
                                    ).sum()
                                )
                                norm_v = np.array(
                                    [N_M_x / norm, N_M_y / norm, N_M_z / norm],
                                    dtype=np.float32,
                                )
                                significative_poins.append([xi, yj, zk])
                                normal_points.append(norm_v)

    return np.array(significative_poins), np.array(normal_points)


def rot_matrices(angle):

    ROTx = np.array(
        [
            [1, 0, 0],
            [0, math.cos(angle[0]), -math.sin(angle[0])],
            [0, math.sin(angle[0]), math.cos(angle[0])],
        ],
        dtype=np.float32,
    )

    ROTy = np.array(
        [
            [math.cos(angle[1]), 0, math.sin(angle[1])],
            [0, 1, 0],
            [-math.sin(angle[1]), 0, math.cos(angle[1])],
        ],
        dtype=np.float32,
    )

    ROTz = np.array(
        [
            [math.cos(angle[2]), -math.sin(angle[2]), 0],
            [math.sin(angle[2]), math.cos(angle[2]), 0],
            [0, 0, 1],
        ],
        dtype=np.float32,
    )

    return ROTx, ROTy, ROTz


def rotate(
    XYZ,
    norm_XYZ,
    norm_light_XYZ,
    angle=np.array([0, 0, 0], dtype=np.float32),
    light_angle=np.array([0, 0, 0], dtype=np.float32),
):

    ROTx, ROTy, ROTz = rot_matrices(angle)

    TOT_ROT = ROTz @ ROTy @ ROTx

    new_coordxyz = XYZ @ TOT_ROT

    new_normxyz = norm_XYZ @ TOT_ROT

    ROTx, ROTy, ROTz = rot_matrices(light_angle)

    new_norm_light = (norm_light_XYZ.T @ (ROTz @ ROTy @ ROTx)).T

    # Sort the array based on the elements in the 2nd column
    sorted_indices = np.argsort(new_coordxyz[:, 2])

    # Use the sorted indices to reorder the array
    sort_xyz = new_coordxyz[sorted_indices]  # i see only nearest z
    sort_normxyz = new_normxyz[sorted_indices]

    # np.vstack([new_coordxyz, new_normxyz]).sort(axis=1, order=)
    return sort_xyz, sort_normxyz, new_norm_light


curr = 0  # or 17 for smooth
dirx = 1
diry = 1


def change_color_ansi_smooth():

    global curr
    global dirx
    global diry

    if dirx > 0:
        curr = curr + 36 * diry

        if curr > 231 or curr < 17:
            curr = curr - 36 * diry + dirx
            diry *= -1

            if curr == 202:
                curr = curr - 2 * dirx - 36
                dirx *= -1

    else:
        curr = curr - 36 * diry

        if curr < 17 or curr > 231:
            curr = curr + 36 * diry + dirx
            diry *= -1

            if curr == 16:
                curr = curr - 2 * dirx + 36
                dirx *= -1

    return f"\033[38;5;214m\033[48;5;{curr}m"


ANSI_PALETTE = [
    "\033[38;5;214m\033[48;5;17m",
    "\033[38;5;1m\033[48;5;10m",
    "\033[38;5;10m\033[48;5;5m",
    "\033[38;5;0m\033[48;5;7m",
]  # use high contrast colors


def change_color_ansi_palette():
    global curr
    curr = (curr + 1) % len(ANSI_PALETTE)
    return ANSI_PALETTE[curr]

COLOR_EFF = [change_color_ansi_palette]
INIT_CURR = [0]
effect = 0
pos = 0

def fast_out(str):
    global curr, tick_time, lock, effect, pos
    # \n useful to make not the terminal update flashing
    # join is faster to concatenate lot of strings

    
    if 1 in tick_time:
        effect = (effect+1)%len(COLOR_EFF)
        curr = INIT_CURR[effect]
        pos = (pos+1)%18

    if 4 in tick_time:
        print(f"\033c{COLOR_EFF[effect]()}" + str, flush=True)

    else:
        print(f"\033c\033[38;5;214m" + str, flush=True)

    lock.acquire()
    tick_time.clear()
    lock.release()


# NOTE: to optimize
def calculate_rays(light_dir, new_XYZ):

    # light form a cylinder of eps from point
    keep_points = []

    indeces_order = np.argsort(
        (new_XYZ @ light_dir).flatten()
    )  # dimension z must be not altered

    most_incident = np.argsort((light_dir).flatten() ** 2)[-1]

    l0, l1, l2 = light_dir[0], light_dir[1], light_dir[2]

    # we calculate which position x,y approzimate to EPS is projected on plane z = OFFSET_Z
    # if match the 2 points are covered by same light beam

    # straight line passing for new_XYZ[index] and parallel to light_dir
    # projected to z=0 is
    # x = t*l0 + x0
    # y = t*l1 + y0
    # z = t*l2 + z0

    if most_incident == 2:
        # -> t = -z0/l2
        t = -new_XYZ[:, 2] / l2
        c1 = t * l0 + new_XYZ[:, 0]
        c2 = t * l1 + new_XYZ[:, 1]

    elif most_incident == 1:
        # -> t = -y0/l1
        t = -new_XYZ[:, 1] / l1
        c1 = t * l0 + new_XYZ[:, 0]
        c2 = t * l2 + new_XYZ[:, 2]

    elif most_incident == 0:
        # -> t = -x0/l0
        t = -new_XYZ[:, 0] / l0
        c1 = t * l1 + new_XYZ[:, 1]
        c2 = t * l2 + new_XYZ[:, 2]

    else:
        print("invalid 0,0,0 light")
        exit()

    # uniform bins (not used)

    # unit_split = int(1/N_UNITS*t.shape[0])

    # c1_sort_idx = np.argsort(c1)
    # c2_sort_idx = np.argsort(c2)

    # for r in range(N_UNITS):
    #    c1[c1_sort_idx[unit_split*r:unit_split*(r+1)]] = r
    #    c2[c2_sort_idx[unit_split*r:unit_split*(r+1)]] = r

    c1 = np.array(c1 // EPS, dtype=np.int32)
    c2 = np.array(c2 // EPS, dtype=np.int32)

    # dictionary approach (less efficient)
    rays_dict = {}


    if TOL > 2:

        for index in indeces_order:  # try to improve with faster dictionaries

            map_pos = (c1[index], c2[index])

            if not map_pos in rays_dict:  # in is slightly faster

                keep_points.append(index)

                rays_dict[map_pos] = 1

            elif rays_dict[map_pos] < TOL:

                keep_points.append(index)

                rays_dict[map_pos] += 1

    else:
        # smart and fast but works only with TOL=1: we need tol to have more regular shapes
        # not effiecient with TOL
        c1min = c1.min()
        c2min = c2.min()

        # Initialize rays_dict
        rays_dict = np.zeros(
            (c1.max() - c1min + 1, c2.max() - c2min + 1, 2), dtype=np.uint32
        )

        c1 -= c1min
        c2 -= c2min
        # Assign values to rays_dict

        rays_dict[c1[indeces_order[::-1]], c2[indeces_order[::-1]], 0] = indeces_order[::-1]
        
        overwritten = indeces_order[rays_dict[c1[indeces_order], c2[indeces_order], 0] != indeces_order]

        rays_dict[c1[overwritten[::-1]], c2[overwritten[::-1]], 1] = overwritten[::-1]


        keep_points = np.unique(rays_dict[c1[indeces_order], c2[indeces_order], :]).tolist()

    keep_points.sort()
    return keep_points


def donut():

    global pos

    # time benchmarks
    time_1 = 0
    time_rot = 0
    time_rays = 0
    time_shadow = 0
    time_out = 0
    # for running mean time
    count = 0

    # keep track of border of figure
    min_x = MAX_CHARS
    max_x = 0
    min_y = MAX_CHARS
    max_y = 0
    # use the border value after 1 complete iteration over all values of rot
    rot = 0
    first_pass = True

    # first generation of the solid
    XYZ, norm_XYZ = gen_xyz()

    while True:

        # print("-1  -1  -1\nv   >    x\n\n1   1   1\n^   <   o")

        x_l = (pos % 3) - 1
        y_l = ((pos // 3) % 3) - 1
        z_l = ((pos // 9) % 3) - 1 #-1 not printed

        if x_l == 0 and y_l == 0 and z_l == 0:
            pos += 1
            continue

        light_array = np.array([[x_l], [y_l], [z_l]], dtype=np.float32)

        norm_light = light_array / math.sqrt((light_array**2).sum())

        for rot in range(PASS):

            start = time.perf_counter()

            new_XYZ, new_norm_XYZ, new_norm_light = rotate(
                XYZ,
                norm_XYZ,
                norm_light,
                angle=np.array(
                    [
                        math.pi * 2 * rot / PASS,
                        -math.pi / 4 + math.pi * 2 * rot / PASS,
                        math.pi * 2 * rot / PASS,
                    ],
                    dtype=np.float32,
                ),
                light_angle=np.array([0, 0, 0], dtype=np.float32),
            )

            t_rays_s = time.perf_counter()
            t_rot = t_rays_s - start

            keep_points = calculate_rays(norm_light, new_XYZ)

            t_shadow_s = time.perf_counter()
            t_rays = t_shadow_s - t_rays_s

            XY_1 = np.vstack(
                [
                    np.array(
                        (new_XYZ[:, 0] * MAX_CHARS) * 1.4 / 0.5 / (-10 + new_XYZ[:, 2])
                        + MAX_CHARS / 2,
                        dtype=np.uint8,
                    ),  # different proportion: terminal not uniform
                    np.array(
                        (new_XYZ[:, 1] * MAX_CHARS) * 2 / 0.5 / (-10 + new_XYZ[:, 2])
                        + MAX_CHARS / 2,
                        dtype=np.uint8,
                    ),
                ]
            ).T

            XY_one_hot = np.zeros((MAX_CHARS, MAX_CHARS), dtype=np.float32)

            keep_points = keep_points[::-1]  # reverse this way is faster

            val = (((new_norm_XYZ @ new_norm_light) ** 2) ** 0.5).squeeze()

            val_shadow = np.zeros(XY_1.shape[0], dtype=np.float32) - 0.0001

            # for idx in range(XY_1.shape[0]): #most of the time

            # if next_idx == idx:

            # val_shadow[idx] = val[idx]
            # try:
            #    next_idx=keep_points.pop()

            # except:
            #    next_idx = -1 #end of ligth points
            # else:
            # XY_one_hot[XY_1[idx,0],XY_1[idx,1]] = -1

            val_shadow[keep_points] = val[keep_points]

            XY_one_hot[XY_1[:, 0], XY_1[:, 1]] = val_shadow

            t_out_s = time.perf_counter()

            t_shadow = t_out_s - t_shadow_s

            padded_XY = np.pad(XY_one_hot, [(1, 1), (1, 1)], mode="constant")
            kernels = np.lib.stride_tricks.sliding_window_view(padded_XY, (3, 3))
            
            k_mean = np.mean(kernels, axis=(2, 3))
            k_count = np.count_nonzero(kernels <= 0, axis=(2, 3))
            k_zero = np.count_nonzero(kernels == 0, axis=(2, 3))

           
            strings = []

            if not first_pass:


                #NOTE: other speedups thanks to
                full_pos = np.where(XY_one_hot != 0)
                only_light = np.where(full_pos and k_count <= 4)

                no_zero_mean = np.zeros(XY_one_hot.shape)
                no_zero_mean[only_light] = k_mean[only_light] * 9 / (9 - k_zero[only_light])

                discrepancy_mean = np.zeros(XY_one_hot.shape)
                discrepancy_mean[only_light] = no_zero_mean[only_light]-XY_one_hot[only_light]

                p1 = np.where(only_light and discrepancy_mean >= 0.2)
                light_1 = np.zeros(XY_one_hot.shape, dtype=np.uint8)
                light_1[p1] = no_zero_mean[p1]*(LIGHT_RANGE*0.9999)+1.0001

                p2_0 = np.where(only_light and discrepancy_mean > -0.4)
                p2 = np.where(p2_0 and discrepancy_mean < 0.2)
                light_2 = np.zeros(XY_one_hot.shape, dtype=np.uint8)
                light_2[p2] = (XY_one_hot[p2]+no_zero_mean[p2])*(LIGHT_RANGE*0.5*0.9999)+1.0001

                #important thing is not to do lot of iterations, not to print less

                width_print = max_y-min_y+1
                height_print = max_x-min_x+1
                width_term, height_term = os.get_terminal_size() 

                padding_x = (width_term - width_print*3 ) // 4
                rest_x = int((width_term - width_print*3 ) - padding_x*4)
                padding_y = (height_term - height_print) // 2 + 1

                # Initialize an empty list to hold the strings
                temp_strings = []

                # Calculate the number of empty lines before and after the main content
                empty_lines = ((width_print*3+padding_x*4+rest_x)*" "+"\n")* padding_y

                # Loop through rows
                for i in range(min_x, max_x + 1):
                    # Initialize an empty string for the current row
                    row_string = ""
                    # Loop through columns
                    for j in range(min_y, max_y + 1):
                        # Check conditions for generating the character for the current position
                        if XY_one_hot[i, j] == 0:
                            char = " "
                        elif k_count[i, j] <= 4:
                            if -0.4 < discrepancy_mean[i, j] < 0.2:
                                char = LIGHTS[light_2[i, j]]
                            elif discrepancy_mean[i, j] <= -0.4:
                                char = "."
                            else:
                                char = LIGHTS[light_1[i, j]]
                        else:
                            char = "."

                        # Append the character to the row string
                        row_string += char

                    # Append the row string to the list of strings
                    temp_strings.append(" "*padding_x+row_string+" "*padding_x+row_string+" "*padding_x+row_string+" "*(padding_x+rest_x)+"\n")

                # Combine the empty lines, row strings, and empty lines
                strings = empty_lines + "".join(temp_strings) + empty_lines

                                



                

            else:

                for i in range(MAX_CHARS):

                    for j in range(MAX_CHARS):

                        if XY_one_hot[i, j] == 0:
                            strings.append(" ")
                        else:

                            # find boundaries of donut (less cost)
                            if max_x < i:
                                max_x = i

                            if min_x > i:
                                min_x = i

                            if max_y < j:
                                max_y = j

                            if min_y > j:
                                min_y = j

                            try:

                                # smoothing and error correction are applied
                                if k_count[i, j] <= 4:

                                    if (k_mean[i, j] * 9 / (9 - k_zero[i, j])- XY_one_hot[i, j]) > 0.2:

                                        light_level = int(
                                            (k_mean[i, j] * 9 / (9 - k_zero[i, j]))
                                            * LIGHT_RANGE
                                            * 0.9999
                                            + 1
                                            + 0.0001
                                        )

                                    elif (
                                        k_mean[i, j] * 9 / (9 - k_zero[i, j])
                                        - XY_one_hot[i, j]
                                    ) < -0.4:

                                        light_level = 0

                                    else:

                                        light_level = int(
                                            (
                                                XY_one_hot[i, j] * 0.5
                                                + k_mean[i, j]
                                                * 9
                                                / (9 - k_zero[i, j])
                                                * 0.5
                                            )
                                            * LIGHT_RANGE
                                            * 0.9999
                                            + 1
                                            + 0.0001
                                        )

                                else:
                                    light_level = 0

                                strings.append(LIGHTS[light_level])

                            except:
                                print("XY_one_hot out of range")
                                print(XY_one_hot.max())
                                print(XY_one_hot.min())
                                exit()

                    strings.append("\n")



            fast_out("".join(strings))

            t_out = time.perf_counter() - t_out_s

            if rot >= PASS - 1:
                rot = rot % PASS
                first_pass = False
                count = 0

            end = time.perf_counter()

            count += 1

            if DEBUG:
                time_1 = time_1 + (end - start - time_1) / count
                time_rot = time_rot + (t_rot - time_rot) / count
                time_rays = time_rays + (t_rays - time_rays) / count
                time_shadow = time_shadow + (t_shadow - time_shadow) / count
                time_out = time_out + (t_out - time_out) / count
                print(f"\nlight:{norm_light.flatten()}\n-1  -1  -1\nv   >    x\n\n1   1   1\n^   <   o\ntotal time:{time_1}\nrot time:{time_rot}\nrays time:{time_rays}\nshadow time:{time_shadow}\nout time:{time_out}", flush = True)
                

            wait_time = 0.0
            if wait_time > 0:
                sleep(wait_time)
