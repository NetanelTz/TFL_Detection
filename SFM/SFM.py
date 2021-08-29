import numpy as np
from math import sqrt


def calc_TFL_dist(prev_container, curr_container, focal, pp):
    norm_prev_pts, norm_curr_pts, R, foe, tZ = prepare_3D_data(prev_container, curr_container, focal, pp)
    if abs(tZ) < 10e-6:
        print('tz = ', tZ)
    elif norm_prev_pts.size == 0:
        print('no prev points')
    elif norm_prev_pts.size == 0:
        print('no curr points')
    else:
        curr_container.corresponding_ind, curr_container.traffic_lights_3d_location, curr_container.valid = calc_3D_data(
            norm_prev_pts, norm_curr_pts, R, foe, tZ)

    return curr_container


def prepare_3D_data(prev_container, curr_container, focal, pp):
    norm_prev_pts = normalize(prev_container.traffic_light, focal, pp)
    norm_curr_pts = normalize(curr_container.traffic_light, focal, pp)
    R, foe, tZ = decompose((curr_container))
    return norm_prev_pts, norm_curr_pts, R, foe, tZ


def calc_3D_data(norm_prev_pts, norm_curr_pts, R, foe, tZ):
    norm_rot_pts = rotate(norm_prev_pts, R)
    pts_3D = []
    corresponding_ind = []
    validVec = []
    for p_curr in norm_curr_pts:
        corresponding_p_ind, corresponding_p_rot = find_corresponding_points(p_curr, norm_rot_pts, foe)
        Z = calc_dist(p_curr, corresponding_p_rot, foe, tZ)
        valid = (Z > 0)
        if not valid:
            Z = 0
        validVec.append(valid)
        P = Z * np.array([p_curr[0], p_curr[1], 1])
        pts_3D.append((P[0], P[1], P[2]))
        corresponding_ind.append(corresponding_p_ind)
    return corresponding_ind, np.array(pts_3D), validVec


def normalize(pts, focal, pp):
    pts_normalize = []
    for pts_x, pts_y in pts:
        l = []
        l.append((pts_x - pp[0]) / focal)
        l.append((pts_y - pp[1]) / focal)
        #      l.append(1)
        pts_normalize.append(l)
    return np.array(pts_normalize)


# transform pixels into normalized pixels using the focal length and principle point

def unnormalize(pts, focal, pp):
    pts_unnormalize = []
    for pts_x, pts_y in pts:
        l = []
        l.append(pts_x * focal + pp[0])
        l.append(pts_y * focal + pp[1])
        # l.append(1)

        pts_unnormalize.append(l)
    return np.array(pts_unnormalize)


def decompose(EM):
    # extract R, foe and tZ from the Ego Motion
    R = EM.EM[:3, :3]
    t = EM.EM[:, 3]
    foe = [t[0] / t[2], t[1] / t[2]]
    return R, foe, t[2]


def rotate(pts, R):
    norm_rotate = []
    for tfl in pts:
        p_vec = np.array([tfl[0], tfl[1], 1])
        result = R.dot(p_vec)
        norm_rotate.append([result[0] / result[2], result[1] / result[2]])
    return np.array(norm_rotate)
    # rotate the points - pts using R


def find_corresponding_points(p, norm_pts_rot, foe):
    m = (foe[1] - p[1]) / (foe[0] - p[0])
    n = (p[1] * foe[0] - foe[1] * p[0]) / (foe[0] - p[0])
    mini = [10]
    for i, point in enumerate(norm_pts_rot):
        x = point[0]
        y = point[1]
        dis = abs((m * x + n - y) / sqrt(pow(m, 2) + 1))
        if dis < mini[0]:
            mini = [dis, i]
    return mini[1], norm_pts_rot[mini[1]]
    # compute the epipolar line between p and foe
    # run over all norm_pts_rot and find the one closest to the epipolar line
    # return the closest point and its index


def calc_dist(p_curr, p_rot, foe, tZ):
    X = (tZ * (foe[0] - p_rot[0])) / (p_curr[0] - p_rot[0])
    Y = (tZ * (foe[1] - p_rot[1])) / (p_curr[1] - p_rot[1])

    snr = (abs(p_curr[0] - p_rot[0]) / abs(p_curr[1] - p_rot[1]))
    if snr > 1:
        return X
    return ((1 - snr) * Y) + (snr * X)

    # (tZ * (0.8 *foe[small] - 0.8 *p_rot[small])) / (0.8 *p_curr[small] - 0.8 *p_rot[small]) + (tZ * (0.2 *foe[big] - 0.2 *p_rot[big])) / (0.2 *p_curr[big] - 0.2 *p_rot[big])
