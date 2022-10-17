# Code modified based on  https://github.com/AI-secure/semantic-randomized-smoothing
import os
import random
import math
import cupy as np
import numpy
import torch
import torchvision
import PIL.Image
from torchvision.transforms import *
import torchvision.transforms.functional as TF
import cv2
import open3d as o3d
import numba
from numba import jit
try:
    from semantic.C import transform_kern as kern
    print('Fast kernel loaded')
    use_kern = True
except Exception:
    print('Fast kernel not available, use PyTorch Kernel')
    use_kern = False

EPS = 1e-6

def filter_frustum(x_ge_0_index, project_positions_flat, project_positions_float, project_positions, points_start, colors):
    project_positions_flat = project_positions_flat[x_ge_0_index]
    project_positions_float = project_positions_float[x_ge_0_index]
    project_positions = project_positions[x_ge_0_index]
    points_start = points_start[x_ge_0_index]
    colors = colors[x_ge_0_index]
    return project_positions_flat, project_positions_float, project_positions, points_start, colors

def find_2d_image(project_positions_flat, project_positions, points_start, colors, intrinsic_matrix):
    point_num = points_start.shape[0]

    # get color image
    h, w = 2 * int(intrinsic_matrix[1][2]), 2 * int(intrinsic_matrix[0][2])
    image = np.ones((h, w, 3))
    dists = np.inf * np.ones((h, w))

    pixel_points = [[[] for j in range(w)] for i in range(h)]
    pixel_closest_point = [[-1 for j in range(w)] for i in range(h)]

    second_dists = np.inf * np.ones((h, w))

    second_pixel_closest_point = [[-1 for j in range(w)] for i in range(h)]
    second_image = np.ones((h, w, 3))

    project_positions[:, :2] = project_positions_flat
    colored_positions = np.hstack([project_positions, colors]).astype(np.float16)
    unique = np.asarray(numpy.unique(np.asnumpy(project_positions_flat), axis=0))
    filtered_list = []

    unique = unique.astype(np.short)
    unique_ = np.repeat(unique[np.newaxis, :], project_positions_flat.shape[0], axis=0)
    project_positions_flat_ = np.repeat(project_positions_flat[:, np.newaxis, :], unique.shape[0], axis=1)
    colored_positions_ = np.repeat(colored_positions[:, np.newaxis, :], unique.shape[0], axis=1)

    same_positions_xy_index = (project_positions_flat_ == unique_)[:, :, 0] & (project_positions_flat_ == unique_)[:, :,
                                                                              1]
    depths_all = np.where(same_positions_xy_index, colored_positions_[:, :, 2], np.inf)

    filtered_positions = colored_positions_[np.argmin(depths_all, axis=0), np.arange(depths_all.shape[1])]

    image[filtered_positions[:, 1:2].astype("int").T[0], filtered_positions[:, :1].astype("int").T[0],
    :] = filtered_positions[:, 3:6]
    return image, second_image, second_pixel_closest_point, pixel_points

def projection_oracle(point_cloud_npy, extrinsic_matrix, intrinsic_matrix):
    point_cloud = point_cloud_npy
    original_positions = point_cloud[:, 0: 3].astype(np.float16)
    colors = point_cloud[:, 3: 6].astype(np.float16)

    positions = np.hstack([original_positions, np.ones((original_positions.shape[0], 1))])
    points_start = (np.linalg.inv(extrinsic_matrix)[0: 3] @ positions.T).T
    project_positions = intrinsic_matrix @ np.linalg.inv(extrinsic_matrix)[0: 3] @ positions.T
    project_positions = project_positions.T
    project_positions_float = project_positions[:, 0:2] / project_positions[:, 2:3]
    project_positions_flat = np.floor(project_positions_float).astype(np.short)

    h, w = 2 * int(intrinsic_matrix[1][2]), 2 * int(intrinsic_matrix[0][2])
    x_ge_0_index = np.where(project_positions_flat[:, 0] >= 0)
    project_positions_flat, project_positions_float, project_positions, points_start, colors = filter_frustum(
        x_ge_0_index, project_positions_flat, project_positions_float, project_positions, points_start, colors)
    y_ge_0_index = np.where(project_positions_flat[:, 1] >= 0)
    project_positions_flat, project_positions_float, project_positions, points_start, colors = filter_frustum(
        y_ge_0_index, project_positions_flat, project_positions_float, project_positions, points_start, colors)
    x_l_w_index = np.where(project_positions_flat[:, 0] < w)
    project_positions_flat, project_positions_float, project_positions, points_start, colors = filter_frustum(
        x_l_w_index, project_positions_flat, project_positions_float, project_positions, points_start, colors)
    y_l_h_index = np.where(project_positions_flat[:, 1] < h)
    project_positions_flat, project_positions_float, project_positions, points_start, colors = filter_frustum(
        y_l_h_index, project_positions_flat, project_positions_float, project_positions, points_start, colors)
    d_g_0_index = np.where(project_positions[:, 2] > 0)
    project_positions_flat, project_positions_float, project_positions, points_start, colors = filter_frustum(
        d_g_0_index, project_positions_flat, project_positions_float, project_positions, points_start, colors)

    return project_positions_flat, project_positions_float, project_positions, points_start, colors

def find_new_extrinsic_matrix(extrinsic_matrix, axis, alpha):
    alpha = np.asnumpy(alpha)
    if axis == 'tz':
        R = np.array([[1, 0, 0],
                      [0, 1, 0],
                      [0, 0, 1]])
        t = np.array([[0, 0, alpha]])
    elif axis == 'tx':
        R = np.array([[1, 0, 0],
                      [0, 1, 0],
                      [0, 0, 1]])
        t = np.array([[alpha, 0, 0]])
    elif axis == 'ty':
        R = np.array([[1, 0, 0],
                      [0, 1, 0],
                      [0, 0, 1]])
        t = np.array([[0, alpha, 0]])
    elif axis == 'rz':
        R = np.array([[numpy.cos(alpha), -numpy.sin(alpha), 0],
                      [numpy.sin(alpha), numpy.cos(alpha), 0],
                      [0, 0, 1]])
        t = np.array([[0, 0, 0]])
    elif axis == 'ry':
        R = np.array([[numpy.cos(alpha), 0, numpy.sin(alpha)],
                      [0, 1, 0],
                      [-numpy.sin(alpha), 0, numpy.cos(alpha)]])
        t = np.array([[0, 0, 0]])
    else:
        R = np.array([[1,0, 0],
                      [0, numpy.cos(alpha), -numpy.sin(alpha)],
                      [0, numpy.sin(alpha), numpy.cos(alpha)]])
        t = np.array([[0, 0, 0]])
    rel_matrix = np.vstack((np.hstack((R, t.T)), np.array([0, 0, 0, 1])))
    return extrinsic_matrix @ rel_matrix

def down_sampling(point_cloud_npy, density, k=-1, k_first=True):
    if density > 0:
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(point_cloud_npy[:, 0: 3])
        point_cloud.colors = o3d.utility.Vector3dVector(point_cloud_npy[:, 3: 6])

        if k > 0:
            if k_first:
                point_cloud = o3d.geometry.PointCloud.uniform_down_sample(point_cloud, k)
                point_cloud = o3d.geometry.PointCloud.voxel_down_sample(point_cloud, density)
            else:
                point_cloud = o3d.geometry.PointCloud.voxel_down_sample(point_cloud, density)
                point_cloud = o3d.geometry.PointCloud.uniform_down_sample(point_cloud, k)
        else:
            point_cloud = o3d.geometry.PointCloud.voxel_down_sample(point_cloud, density)
        original_positions = np.asarray(point_cloud.points)
        colors = np.asarray(point_cloud.colors)
        return np.hstack([original_positions, colors])
    else:
        return np.asnumpy(point_cloud_npy)

class Noise:
    def __init__(self, sigma):
        self.sigma = sigma

    def proc(self, input):
        noise = torch.randn_like(input) * self.sigma
        return input + noise

    def batch_proc(self, inputs):
        noise = torch.randn_like(inputs) * self.sigma
        return inputs + noise


class Rotation:
    def __init__(self, canopy, rotation_angle):
        self.h = canopy.shape[-2]
        self.w = canopy.shape[-1]
        assert self.h == self.w
        self.mask = torch.ones((self.h, self.w))
        for i in range(self.h):
            for j in range(self.w):
                if (i - (self.h-1)/2.0) ** 2 + (j - (self.w-1)/2.0) ** 2 > ((self.h-1)/2.0) ** 2:
                    self.mask[i][j] = 0
        self.mask.unsqueeze_(0)
        self.rotation_angle = rotation_angle

    def gen_param(self):
        return random.uniform(-self.rotation_angle, self.rotation_angle)

    def raw_proc(self, input: torch.Tensor, angle: float):
        if abs(angle) < EPS:
            return input
        if use_kern:
            np_input = np.ascontiguousarray(input.numpy(), dtype=np.float)
            np_output = kern.rotation(np_input, angle)
            output = torch.tensor(np_output)
            return output
        else:
            c, h, w = input.shape
            cy, cx = (h - 1) / 2.0, (w - 1) / 2.0

            rows = torch.linspace(0.0, h - 1, steps=h)
            cols = torch.linspace(0.0, w - 1, steps=w)

            dist_mat = ((rows - cy) * (rows - cy)).unsqueeze(1) + ((cols - cx) * (cols - cx)).unsqueeze(0)
            dist_mat = torch.sqrt(dist_mat)

            rows_mat = rows.unsqueeze(1).repeat(1, w)
            cols_mat = cols.repeat(h, 1)
            alpha_mat = torch.atan2(rows_mat - cy, cols_mat - cx)
            beta_mat = alpha_mat + angle * math.pi / 180.0

            ny_mat, nx_mat = dist_mat * torch.sin(beta_mat) + cy, dist_mat * torch.cos(beta_mat) + cx
            nyl_mat, nxl_mat = torch.floor(ny_mat).type(torch.LongTensor), torch.floor(nx_mat).type(torch.LongTensor)
            nyp_mat, nxp_mat = nyl_mat + 1, nxl_mat + 1
            torch.clamp_(nyl_mat, min=0, max=h - 1)
            torch.clamp_(nxl_mat, min=0, max=w - 1)
            torch.clamp_(nyp_mat, min=0, max=h - 1)
            torch.clamp_(nxp_mat, min=0, max=w - 1)

            nyb_cell, nxb_cell = torch.flatten(nyl_mat), torch.flatten(nxl_mat)
            nyp_cell, nxp_cell = torch.flatten(nyp_mat), torch.flatten(nxp_mat)

            Pll = torch.gather(input.reshape(c, h * w), dim=1, index=(nyb_cell * w + nxb_cell).repeat(c, 1)).reshape(c, h, w)
            Plr = torch.gather(input.reshape(c, h * w), dim=1, index=(nyb_cell * w + nxp_cell).repeat(c, 1)).reshape(c, h, w)
            Prl = torch.gather(input.reshape(c, h * w), dim=1, index=(nyp_cell * w + nxb_cell).repeat(c, 1)).reshape(c, h, w)
            Prr = torch.gather(input.reshape(c, h * w), dim=1, index=(nyp_cell * w + nxp_cell).repeat(c, 1)).reshape(c, h, w)

            nyl_mat, nxl_mat = nyl_mat.type(torch.FloatTensor), nxl_mat.type(torch.FloatTensor)

            raw = (ny_mat - nyl_mat) * (nx_mat - nxl_mat) * Prr + \
                  (ny_mat - nyl_mat) * (1.0 - nx_mat + nxl_mat) * Prl + \
                  (1.0 - ny_mat + nyl_mat) * (nx_mat - nxl_mat) * Plr + \
                  (1.0 - ny_mat + nyl_mat) * (1.0 - nx_mat + nxl_mat) * Pll
            out = raw
            return out

    def old_raw_proc(self, input, angle):
        pil = TF.to_pil_image(input)
        rot = TF.rotate(pil, angle, PIL.Image.BILINEAR)
        out = TF.to_tensor(rot)
        return out

    def masking(self, input: torch.Tensor):
        return input * self.mask

    def proc(self, input: torch.Tensor, angle: float):
        return self.masking(self.raw_proc(input, angle) if abs(angle) > EPS else input)

    def batch_proc(self, inputs):
        outs = torch.zeros_like(inputs)
        for i in range(len(inputs)):
            outs[i] = self.proc(inputs[i], self.gen_param())
        return outs

    def batch_masking(self, inputs):
        return inputs * self.mask.unsqueeze(0)


class Translational:

    def __init__(self, canopy, sigma):
        self.sigma = sigma
        self.c, self.h, self.w = canopy.shape

    def gen_param(self):
        tx, ty = torch.randn(2)
        tx, ty = tx.item(), ty.item()
        return tx * self.sigma, ty * self.sigma

    def proc(self, input, dx, dy):
        nx, ny = round(dx), round(dy)
        nx, ny = nx % self.h, ny % self.w
        out = torch.zeros_like(input)
        if nx > 0 and ny > 0:
            out[:, -nx:, -ny:] = input[:, :nx, :ny]
            out[:, -nx:, :-ny] = input[:, :nx, ny:]
            out[:, :-nx, -ny:] = input[:, nx:, :ny]
            out[:, :-nx, :-ny] = input[:, nx:, ny:]
        elif ny > 0:
            out[:, :, -ny:] = input[:, :, :ny]
            out[:, :, :-ny] = input[:, :, ny:]
        elif nx > 0:
            out[:, -nx:, :] = input[:, :nx, :]
            out[:, :-nx, :] = input[:, nx:, :]
        else:
            out = input
        return out

    def batch_proc(self, inputs):
        outs = torch.zeros_like(inputs)
        for i in range(len(inputs)):
            outs[i] = self.proc(inputs[i], *self.gen_param())
        return outs

class ResolvableProjection:

    def __init__(self, canopy, sigma, axis):
        self.sigma = sigma
        self.intrinsic_matrix = canopy["intrinsic_matrix"]
        self.intrinsic_matrix[0][0] /= 8
        self.intrinsic_matrix[1][1] /= 8
        self.intrinsic_matrix[0][2] /= 8
        self.intrinsic_matrix[1][2] /= 8
        self.axis = axis

        self.density_dic = {"ty": 0.0137, "tx": 0.01365,"tz": 0.0133,"ry": 0.0134,"rx": 0.01355,"rz": 0.0135}
        self.k_dic = {"ty": 7,"tx": 6,"tz": 7,"ry": 7,"rx": 6,"rz": 7}
        self.k_first_dic = {"ty": True, "tx": True,"tz": True,"ry": True,"rx": True,"rz": True}

    def proc(self, input, empirical=False, type=None):
        # print("#############", input)
        self.extrinsic_matrix = input["pose"]
        # self.complete_3D_oracle = down_sampling(input["point_cloud"], self.density_dic[self.axis], self.k_dic[self.axis],
        #                                         self.k_first_dic[self.axis])
        self.complete_3D_oracle = input["point_cloud"]#[numpy.delete(numpy.arange(0, input["point_cloud"].shape[0]), numpy.arange(0, input["point_cloud"].shape[0], 5), None)]
        if empirical:
            if type == "uniform":
                alpha = 2 * (np.random.uniform()-0.5) * self.sigma
            elif type == "beta":
                alpha = 2 * (np.random.beta(0.5,0.5) - 0.5) * self.sigma
            else:
                assert type == "benign"
                alpha = 0.0 * self.sigma
        else:
            alpha = np.random.normal() * self.sigma
        extrinsic_matrix = find_new_extrinsic_matrix(self.extrinsic_matrix, self.axis, alpha)
        project_positions_flat, project_positions_float, project_positions, points_start, colors = projection_oracle(
            self.complete_3D_oracle, extrinsic_matrix, self.intrinsic_matrix)
        image, second_image, second_pixel_closest_point, pixel_points = find_2d_image(project_positions_flat,
                                                                                      project_positions, points_start,
                                                                                      colors, self.intrinsic_matrix)
        return torch.as_tensor(image, device='cuda')

    def pertubate(self, input, empirical=False, type=None):
        # print("#############", input)
        output = input.copy()
        extrinsic_matrix = input["pose"]
        # self.complete_3D_oracle = down_sampling(input["point_cloud"], self.density_dic[self.axis], self.k_dic[self.axis],
        #                                         self.k_first_dic[self.axis])
        # self.complete_3D_oracle = input["point_cloud"]#[numpy.delete(numpy.arange(0, input["point_cloud"].shape[0]), numpy.arange(0, input["point_cloud"].shape[0], 5), None)]
        if empirical:
            if type == "uniform":
                alpha = 2 * (np.random.uniform()-0.5) * self.sigma
            elif type == "beta":
                alpha = 2 * (np.random.beta(0.5,0.5) - 0.5) * self.sigma
            else:
                assert type == "benign"
                alpha = 0.0 * self.sigma
        else:
            alpha = np.random.normal() * self.sigma
        output["pose"] = find_new_extrinsic_matrix(extrinsic_matrix, self.axis, alpha)

        return output

    def batch_proc(self, inputs, empirical=False, type=None):
        outs = [] #torch.zeros_like(inputs)
        # print("##########", len(inputs))
        for i in range(len(inputs)):
            outs.append(self.proc(inputs[i], empirical, type))
        return outs

class DiffResolvableProjection:

    def __init__(self, canopy, axis, range=0):
        self.intrinsic_matrix = canopy["intrinsic_matrix"]
        self.intrinsic_matrix[0][0] /= 8
        self.intrinsic_matrix[1][1] /= 8
        self.intrinsic_matrix[0][2] /= 8
        self.intrinsic_matrix[1][2] /= 8
        self.axis = axis
        self.range = range
        self.density_dic = {"ty": 0.0137, "tx": 0.01365,"tz": 0.0133,"ry": 0.0134,"rx": 0.01355,"rz": 0.0135}
        self.k_dic = {"ty": 7,"tx": 6,"tz": 7,"ry": 7,"rx": 6,"rz": 7}
        self.k_first_dic = {"ty": True, "tx": True,"tz": True,"ry": True,"rx": True,"rz": True}

    def proc(self, input, alpha=None):
        self.extrinsic_matrix = input["pose"]
        # self.complete_3D_oracle = down_sampling(input["point_cloud"], self.density_dic[self.axis], self.k_dic[self.axis],
        #                                         self.k_first_dic[self.axis])
        self.complete_3D_oracle = input["point_cloud"]#[numpy.delete(numpy.arange(0, input["point_cloud"].shape[0]), numpy.arange(0, input["point_cloud"].shape[0], 5), None)]

        if alpha == None:
            alpha = np.random.uniform(-self.range, self.range)
        extrinsic_matrix = find_new_extrinsic_matrix(self.extrinsic_matrix, self.axis, alpha)
        project_positions_flat, project_positions_float, project_positions, points_start, colors = projection_oracle(
            self.complete_3D_oracle, extrinsic_matrix, self.intrinsic_matrix)
        image, second_image, second_pixel_closest_point, pixel_points = find_2d_image(project_positions_flat,
                                                                                      project_positions, points_start,
                                                                                      colors, self.intrinsic_matrix)
        return torch.as_tensor(image, device='cuda')

    def batch_proc(self, inputs):
        outs = [] #torch.zeros_like(inputs)
        for i in range(len(inputs)):
            outs.append(self.proc(inputs[i]))
        return outs


class BlackTranslational(Translational):

    def __init__(self, canopy, sigma):
        super(BlackTranslational, self).__init__(canopy, sigma)

    def proc(self, input, dx, dy):
        nx, ny = round(dx), round(dy)
        out = torch.zeros_like(input)
        nx = nx % self.h if nx > 0 else nx % (-self.h)
        ny = ny % self.w if ny > 0 else ny % (-self.w)
        if nx > 0 and ny > 0:
            out[:, :-nx, :-ny] = input[:, nx:, ny:]
        elif nx > 0 and ny == 0:
            out[:, :-nx, :] = input[:, nx:, :]
        elif nx > 0 and ny < 0:
            out[:, :-nx, -ny:] = input[:, nx:, :ny]
        elif nx == 0 and ny > 0:
            out[:, :, :-ny] = input[:, :, ny:]
        elif nx == 0 and ny == 0:
            out = input
        elif nx == 0 and ny < 0:
            out[:, :, -ny:] = input[:, :, :ny]
        elif nx < 0 and ny > 0:
            out[:, -nx:, :-ny] = input[:, :nx, ny:]
        elif nx < 0 and ny == 0:
            out[:, -nx:, :] = input[:, :nx, :]
        elif nx < 0 and ny < 0:
            out[:, -nx:, -ny:] = input[:, :nx, :ny]
        return out



class BrightnessShift:

    def __init__(self, sigma):
        self.sigma = sigma

    def gen_param(self):
        d = torch.randn(1).item() * self.sigma
        return d

    def proc(self, input, d):
        # print(d)
        return input + d

    def batch_proc(self, inputs):
        outs = torch.zeros_like(inputs)
        for i in range(len(inputs)):
            outs[i] = self.proc(inputs[i], self.gen_param())
        return outs


class BrightnessScale:

    def __init__(self, sigma):
        self.sigma = sigma

    def gen_param(self):
        d = torch.randn(1).item() * self.sigma
        return d

    def proc(self, input, dk):
        # scale by exp(dk)
        # print(dk)
        return input * math.exp(dk)

    def batch_proc(self, inputs):
        outs = torch.zeros_like(inputs)
        for i in range(len(inputs)):
            outs[i] = self.proc(inputs[i], self.gen_param())
        return outs


class Resize:

    def __init__(self, canopy, sl, sr):
        self.sl, self.sr = sl, sr
        self.c, self.h, self.w = canopy.shape
        self.rows = torch.linspace(0.0, self.h - 1, steps=self.h)
        self.cols = torch.linspace(0.0, self.w - 1, steps=self.w)


    def gen_param(self):
        return random.uniform(self.sl, self.sr)

    def proc(self, input, s):
        if abs(s - 1) < EPS:
            return input
        if use_kern:
            np_input = np.ascontiguousarray(input.numpy(), dtype=np.float)
            np_output = kern.scaling(np_input, s)
            output = torch.tensor(np_output)
            return output
        else:
            c, h, w = self.c, self.h, self.w
            cy, cx = float(h - 1) / 2.0, float(w - 1) / 2.0
            nys = (self.rows - cy) / s + cy
            nxs = (self.cols - cx) / s + cx

            nysl, nxsl = torch.floor(nys), torch.floor(nxs)
            nysr, nxsr = nysl + 1, nxsl + 1

            nysl = nysl.clamp(min=0, max=h-1).type(torch.LongTensor)
            nxsl = nxsl.clamp(min=0, max=w-1).type(torch.LongTensor)
            nysr = nysr.clamp(min=0, max=h-1).type(torch.LongTensor)
            nxsr = nxsr.clamp(min=0, max=w-1).type(torch.LongTensor)

            nyl_mat, nyr_mat, ny_mat = nysl.unsqueeze(1).repeat(1, w), nysr.unsqueeze(1).repeat(1, w), nys.unsqueeze(1).repeat(1, w)
            nxl_mat, nxr_mat, nx_mat = nxsl.repeat(h, 1), nxsr.repeat(h, 1), nxs.repeat(h, 1)

            nyl_arr, nyr_arr, nxl_arr, nxr_arr = nyl_mat.flatten(), nyr_mat.flatten(), nxl_mat.flatten(), nxr_mat.flatten()

            imgymin = max(math.ceil((1 - s) * cy), 0)
            imgymax = min(math.floor((1 - s) * cy + s * (h - 1)), h - 1)
            imgxmin = max(math.ceil((1 - s) * cx), 0)
            imgxmax = min(math.floor((1 - s) * cx + s * (h - 1)), w - 1)

            # Pll_old = torch.gather(torch.index_select(input, dim=1, index=nyl_arr), dim=2,
            #                        index=nxl_arr.repeat(c, 1).unsqueeze(2)).reshape(c, h, w)
            # Plr_old = torch.gather(torch.index_select(input, dim=1, index=nyl_arr), dim=2,
            #                        index=nxr_arr.repeat(c, 1).unsqueeze(2)).reshape(c, h, w)
            # Prl_old = torch.gather(torch.index_select(input, dim=1, index=nyr_arr), dim=2,
            #                        index=nxl_arr.repeat(c, 1).unsqueeze(2)).reshape(c, h, w)
            # Prr_old = torch.gather(torch.index_select(input, dim=1, index=nyr_arr), dim=2,
            #                        index=nxr_arr.repeat(c, 1).unsqueeze(2)).reshape(c, h, w)

            Pll = torch.gather(input.reshape(c, h * w), dim=1, index=(nxl_arr + nyl_arr * w).repeat(c, 1)).reshape(c, h, w)
            Plr = torch.gather(input.reshape(c, h * w), dim=1, index=(nxr_arr + nyl_arr * w).repeat(c, 1)).reshape(c, h, w)
            Prl = torch.gather(input.reshape(c, h * w), dim=1, index=(nxl_arr + nyr_arr * w).repeat(c, 1)).reshape(c, h, w)
            Prr = torch.gather(input.reshape(c, h * w), dim=1, index=(nxr_arr + nyr_arr * w).repeat(c, 1)).reshape(c, h, w)

            # print(torch.sum(torch.abs(Pll - Pll_old)))
            # print(torch.sum(torch.abs(Plr - Plr_old)))
            # print(torch.sum(torch.abs(Prl - Prl_old)))
            # print(torch.sum(torch.abs(Prr - Prr_old)))

            nxl_mat, nyl_mat = nxl_mat.type(torch.FloatTensor), nyl_mat.type(torch.FloatTensor)

            out = torch.zeros_like(input)
            out[:, imgymin: imgymax + 1, imgxmin: imgxmax + 1] = (
                (ny_mat - nyl_mat) * (nx_mat - nxl_mat) * Prr +
                (1.0 - ny_mat + nyl_mat) * (nx_mat - nxl_mat) * Plr +
                (ny_mat - nyl_mat) * (1.0 - nx_mat + nxl_mat) * Prl +
                (1.0 - ny_mat + nyl_mat) * (1.0 - nx_mat + nxl_mat) * Pll)[:, imgymin: imgymax + 1, imgxmin: imgxmax + 1]

            return out

    def batch_proc(self, inputs):
        outs = torch.zeros_like(inputs)
        for i in range(len(inputs)):
            outs[i] = self.proc(inputs[i], self.gen_param())
        return outs


class Gaussian:
    # it adopts uniform distribution
    def __init__(self, sigma):
        self.sigma = sigma
        self.sigma2 = sigma ** 2.0

    def gen_param(self):
        r = random.uniform(0.0, self.sigma2)
        return r

    def proc(self, input, r2):
        if (abs(r2) < 1e-6):
            return input
        out = cv2.GaussianBlur(input.numpy().transpose(1, 2, 0), (0, 0), math.sqrt(r2), borderType=cv2.BORDER_REFLECT101)
        if out.ndim == 2:
            out = np.expand_dims(out, 2)
        out = torch.from_numpy(out.transpose(2, 0, 1))
        return out

    def batch_proc(self, inputs):
        outs = torch.zeros_like(inputs)
        for i in range(len(inputs)):
            outs[i] = self.proc(inputs[i], self.gen_param())
        return outs

class ExpGaussian(Gaussian):
    # it adopts exponential distribution
    # where the sigma is actually lambda in exponential distribution Exp(1/lambda)
    def __init__(self, sigma):
        super(ExpGaussian, self).__init__(sigma)

    def gen_param(self):
        r = - self.sigma * math.log(random.uniform(0.0, 1.0))
        return r

class FoldGaussian(Gaussian):
    def __init__(self, sigma):
        super(FoldGaussian, self).__init__(sigma)

    def gen_param(self):
        r = abs(random.normalvariate(0.0, self.sigma))
        return r


def visualize(img, outfile):
    img = torch.tensor(img).clamp_(min=0.0, max=1.0)
    if not os.path.exists(os.path.dirname(outfile)):
        os.makedirs(os.path.dirname(outfile))
    torchvision.utils.save_image(img, outfile, range=(0.0, 1.0))


