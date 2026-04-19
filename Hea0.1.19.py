import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error
import itertools
import os
import math
import shutil
import re
from scipy.io import loadmat
import cv2

import warnings

warnings.filterwarnings('ignore', category=UserWarning)
import sympy as sp
import random
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm, trange

from web_export import ExportPayload, export_prediction_bundle

# 在v0.2中(即当前版本）应当完成数值方法使用固定神经网络CNN替代的工作，
# 而后在v0.2版本的基础上应该添加神经网络模块


interval = 1

mse0 = 0.0


def set_random_seed(mySeed=0):
    torch.manual_seed(mySeed)
    torch.cuda.manual_seed(mySeed)
    torch.cuda.manual_seed_all(mySeed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(mySeed)
    random.seed(mySeed)


# seed = np.random.randint(1,100000)
# seed = 1
# set_random_seed(mySeed=seed)  # mySeed = 4,95.5左右
# print('using seed:', seed)

torch.set_default_tensor_type(torch.FloatTensor)


# torch.autograd.set_detect_anomaly(True)

class NetResTransformer(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, patch_size, num_heads, num_layers, kernel_size,
                 stride, padding):
        super(NetResTransformer, self).__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.num_layers = num_layers

        # Patch embedding
        self.patch_embed = nn.Conv2d(in_channels, hidden_channels, kernel_size=patch_size, stride=patch_size)

        # Transformer Encoder layers
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_channels, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Final output layer
        self.fc_out = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # Step 1: Convert image to patches (flattening the spatial dimensions)
        _, _, HH, WW = x.shape

        x = self.patch_embed(x)  # [batch_size, hidden_channels, H', W']

        # Reshape to [batch_size, seq_len, hidden_channels] for Transformer
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # [batch_size, seq_len, hidden_channels] where seq_len = H' * W'

        # Step 2: Transformer processing
        x = self.transformer_encoder(x)

        # Step 3: Project back to the original size (using 1x1 conv to match output channels)
        x = x.transpose(1, 2).reshape(B, C, H, W)  # [batch_size, hidden_channels, H', W']

        # Step 4: Output layer
        x = self.fc_out(x)  # [batch_size, out_channels, H', W']

        # Step 5: Upsampling to match the input size if needed
        # x = F.interpolate(x, size=(HH, WW), mode='bilinear', align_corners=False)

        x = F.interpolate(x, size=(HH, WW), mode='nearest')

        return x


class NetRes(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, kernel_size, stride, padding):
        super(NetRes, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size, stride=stride, padding=padding)
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size, stride=stride, padding=padding)
        self.conv3 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size, stride=stride, padding=padding)
        self.conv4 = nn.Conv2d(hidden_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.activation = nn.Tanh()

    def forward(self, x):
        # print('before:', x.shape)                         # before: torch.Size([1, 1, 100, 100])
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x))
        x = self.activation(self.conv4(x))
        # print('after:', x.shape)                          # after: torch.Size([1, 1, 100, 100])
        return x


class Conv1dDerivativeCN(nn.Module):
    def __init__(self, DerFilter, deno, kernel_size=3, name=''):
        super(Conv1dDerivativeCN, self).__init__()

        self.deno = deno  # $\delta$*constant in the finite difference
        self.name = name
        self.input_channels = 1
        self.output_channels = 1
        self.kernel_size = kernel_size

        self.padding = int((kernel_size - 1) / 2)
        self.filter = nn.Conv1d(self.input_channels, self.output_channels, self.kernel_size,
                                1, padding=1, bias=False)

        # Fixed gradient operator
        self.filter.weight = nn.Parameter(torch.tensor(DerFilter, dtype=torch.float32), requires_grad=False)

        self.filter = self.filter.cuda()

    def forward(self, input):
        derivative = self.filter(input)
        return derivative / self.deno


class Conv1dDerivative(nn.Module):
    def __init__(self, DerFilter, deno, kernel_size=3, name=''):
        super(Conv1dDerivative, self).__init__()

        self.deno = deno  # $\delta$*constant in the finite difference
        self.name = name
        self.input_channels = 1
        self.output_channels = 1
        self.kernel_size = kernel_size

        self.padding = int((kernel_size - 1) / 2)
        self.filter = nn.Conv1d(self.input_channels, self.output_channels, self.kernel_size,
                                1, padding=2, bias=False)

        # Fixed gradient operator
        self.filter.weight = nn.Parameter(torch.tensor(DerFilter, dtype=torch.float32), requires_grad=False)

        self.filter = self.filter.cuda()

    def forward(self, input):
        derivative = self.filter(input)
        return derivative / self.deno


class Conv2dDerivative(nn.Module):
    def __init__(self, DerFilter, deno, kernel_size=5, name=''):
        '''
        :param DerFilter: constructed derivative filter, e.g. Laplace filter
        :param deno: resolution of the filter, used to divide the output, e.g. c*dt, c*dx or c*dx^2
        :param kernel_size:
        :param name: optional name for the operator
        '''
        super(Conv2dDerivative, self).__init__()
        self.deno = deno  # constant in the finite difference
        self.name = name
        self.input_channels = 1
        self.output_channels = 1
        self.kernel_size = kernel_size

        self.padding = int((kernel_size - 1) / 2)
        self.filter = nn.Conv2d(self.input_channels, self.output_channels, self.kernel_size,
                                1, padding=1, bias=False)
        # Fixed gradient operator
        self.filter.weight = nn.Parameter(torch.tensor(DerFilter, dtype=torch.float32), requires_grad=False)
        self.filter = self.filter.cuda()

    def forward(self, input):
        derivative = self.filter(input)
        return derivative / self.deno


class PRNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.nx = 101  # grid points in x-Direction
        self.ny = 101  # grid points in y-Direction
        self.nu = 1.  # viscosity

        # self.dx = self.dy = 0.02
        self.dx = 1. / self.nx
        self.dy = 1. / self.ny
        self.dd = self.dx

        self.cfl = 0.25

        # self.dt = 0.002
        # self.dt = self.cfl * self.dx * self.dy / self.nu  # time step based on von neumann stability analysis
        # self.nt = 1000
        # self.dt = 0.2
        # self.dt = 0.00001
        self.dt = 1e-5

        print('dt:', self.dt)

        self.step = 50
        self.long_step = 100
        self.save_step = 100
        self.print_step = 10

        self.batch_size = 1
        self.in_channels = 1
        self.hidden_channels = 20
        # self.height = 50
        # self.width = 50
        self.out_channels = 1
        self.kernel_size = 3
        self.stride = 1

        self.hw = 101

        # self.cycle = 10
        self.cycle = 50
        self.single_step = self.step // self.cycle
        self.all_fusion = False

        self.ndt = self.single_step * self.dt

        # self.xlin = np.linspace(0, 2, self.nx)
        # self.ylin = np.linspace(0, 2, self.ny)
        #
        # print('xlin:', self.xlin)
        # print('xlin:', self.ylin)

        self.u0 = self.getu0bysympy()

        # self.u0 = np.zeros((self.ny, self.nx))
        # self.u0[int(0.8 / self.dy):int(1.2 / self.dy + 1) - 1, int(0.8 / self.dx):int(1.2 / self.dx + 1) - 1] = 1

        gt = self.getgtsympy(700)
        # gt = self.getgtsympy(10000)
        # for i in range(500):
        #     plt.clf()
        #     plt.imshow(gt[i, ...], cmap='hot', vmin=0, vmax=1)
        #     plt.pause(0.2)
        #     plt.ioff()

        self.gt = gt[:self.step + 1, ...]
        self.long_gt = gt[:self.long_step + 1, ...]

        # self.gt = loadmat('heatsolution_dt0.002_n100t100_nu0.05_totaltime_0.4_xi.mat')['U_matrix'][:self.step + 1, ...]
        # self.long_gt = loadmat('heatsolution_dt0.002_n100t100_nu0.05_totaltime_2_xi.mat')['U_matrix'][:self.long_step + 1, ...]

        print('self.gt:', self.gt.shape)
        print('self.long_gt:', self.long_gt.shape)

        self.getnn()

        self.current_epoch = ''

        self.MSE0 = 0.
        self.frameMSE = 0.

        self.MSEcutu00 = 0.

        self.long_MSE0 = 0.
        self.long_MSE0cutu0 = 0.
        self.long_frameMSE = 0.

        self.long_percnn_mse = 0.
        # print('self.long_percnn_result_mse', self.long_percnn_mse.shape)

        # self.long_percnn_mse = np.load('percnn_result_mse.npy')
        # print('self.long_percnn_result_mse', self.long_percnn_mse.shape)

        self.A = self.build_sparse_matrix().cuda()

        self.intervel = interval

    def getgtsympy(self, n, delt=1e-5):
        # 定义符号变量
        x, y, t = sp.symbols('x y t')

        # 定义解析解函数
        u_exact = sp.exp(-2 * sp.pi ** 2 * t) * sp.sin(sp.pi * x) * sp.sin(sp.pi * y)
        # print("解析解表达式:")
        # display(u_exact)  # 以 LaTeX 格式显示公式

        # 将符号表达式转换为数值计算函数
        u_numeric = sp.lambdify((x, y, t), u_exact, "numpy")

        # 设置空间网格
        xx = np.linspace(0, 1, self.hw)
        yy = np.linspace(0, 1, self.hw)
        X, Y = np.meshgrid(xx, yy)

        result = []
        for i in range(n):
            u = u_numeric(X, Y, i * delt)
            result.append(u.copy())
        result = np.array(result)
        # print('result:', result.shape)
        return result

    def getu0bysympy(self):
        # 定义符号变量
        x, y, t = sp.symbols('x y t')

        # 定义解析解函数
        u_exact = sp.exp(-2 * sp.pi ** 2 * t) * sp.sin(sp.pi * x) * sp.sin(sp.pi * y)
        print("解析解表达式:")
        # display(u_exact)  # 以 LaTeX 格式显示公式

        # 将符号表达式转换为数值计算函数
        u_numeric = sp.lambdify((x, y, t), u_exact, "numpy")

        # 设置空间网格
        xx = np.linspace(0, 1, self.hw)
        yy = np.linspace(0, 1, self.hw)
        X, Y = np.meshgrid(xx, yy)

        u0 = u_numeric(X, Y, 0)
        return u0

    def getnn(self):
        # Calculate padding size to keep the feature map size unchanged
        self.padding = self.calculate_padding(self.kernel_size)

        # self.model = nn.Sequential(
        #     nn.Conv2d(self.in_channels, self.hidden_channels, self.kernel_size, stride=self.stride, padding=self.padding),
        #     nn.Tanh(),
        #     nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, stride=self.stride,
        #               padding=self.padding),
        #     nn.Tanh(),
        #     nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, stride=self.stride,
        #               padding=self.padding),
        #     nn.Tanh(),
        #     nn.Conv2d(self.hidden_channels, self.out_channels, self.kernel_size, stride=self.stride,
        #               padding=self.padding),
        #     nn.Tanh(),
        # ).cuda()

        # 1
        # self.model = NetResTransformer(in_channels=1, hidden_channels=64, out_channels=1, patch_size=8, num_heads=4,
        #                           num_layers=4, kernel_size=3, stride=1, padding=1).cuda()

        # 2
        # self.model = NetResTransformer(in_channels=1, hidden_channels=64, out_channels=1, patch_size=8, num_heads=8,
        #                                num_layers=4, kernel_size=3, stride=1, padding=1).cuda()

        # 3
        # self.model = NetResTransformer(in_channels=1, hidden_channels=64, out_channels=1, patch_size=8, num_heads=2,
        #                                num_layers=4, kernel_size=3, stride=1, padding=1).cuda()

        # 4
        # self.model = NetResTransformer(in_channels=1, hidden_channels=64, out_channels=1, patch_size=4, num_heads=4,
        #                                num_layers=4, kernel_size=3, stride=1, padding=1).cuda()

        # 5
        # self.model = NetResTransformer(in_channels=1, hidden_channels=64, out_channels=1, patch_size=16, num_heads=4,
        #                                num_layers=4, kernel_size=3, stride=1, padding=1).cuda()

        self.model = NetResTransformer(in_channels=1, hidden_channels=128, out_channels=1, patch_size=16, num_heads=4,
                                       num_layers=4, kernel_size=3, stride=1, padding=1).cuda()

        # self.model = NetRes(self.in_channels, self.hidden_channels, self.out_channels, self.kernel_size, self.stride, self.padding).cuda()

        self.A = 1e-3 * torch.ones(size=(self.step + 1, 3, self.hw, self.hw)).cuda()
        self.ratio = torch.nn.Parameter(torch.tensor(1e-32, requires_grad=True))
        # self.ratio = torch.nn.Parameter(torch.tensor(-1e1, requires_grad=True))
        self.optim = torch.optim.Adam(itertools.chain(self.model.parameters(), [self.ratio]), lr=1e-8)
        self.loss_fn = torch.nn.MSELoss()
        # self.scheduler = torch.optim.lr_scheduler.StepLR(self.optim, step_size=5000, gamma=0.5)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optim, gamma=1. - 1e-3)

        self.init_weights_normal(self.model)

    def data_driven_PR_forward(self, u):  # 64, 1, 100, 100
        un = u.clone()
        u = un + (self.nu * self.dt / self.dd ** 2) * self.data_driven_lap(un)
        return u

    def PR_forward(self, u0, n):
        u0 = u0[0, 0, ...]
        result = []
        # result.append(torch.tensor(u0.copy(), dtype=torch.float32))
        result.append(torch.tensor(u0.clone().cuda()))
        u = torch.tensor(u0).cuda()
        for n in range(n):
            un = u.clone()

            u = un + (self.nu * self.dt / self.dd ** 2) * self.lap(un)

            # u[0, :] = 1
            # u[-1, :] = 1
            # u[:, 0] = 1
            # u[:, -1] = 1

            result.append(u.clone())

        # (1001, 100, 100)
        # print('result:', result)
        # result = torch.tensor(result)

        result = torch.stack(tuple(result), dim=0)
        result = result[:, None, ...]
        # print('result:', result.shape)

        # rsl = result.detach().cpu().numpy()
        # print('rsl:', rsl.shape)
        # for i in range(21):
        #     plt.clf()
        #     plt.imshow(rsl[i, 0, ...], cmap='hot')
        #     plt.pause(0.5)
        #     plt.ioff()
        # plt.close()

        return result

    def data_driven_NN_forward(self, u):
        result = self.model(u)
        return result

    def NN_forward(self, U0, m):

        # U0 = torch.stack((u_prev, v_prev, p_prev), dim=0).unsqueeze(0)
        # print('U0:', U0.shape)

        result_U = []

        # result_U.append(torch.tensor(U0).unsqueeze(0).unsqueeze(0).cuda())
        # u = torch.tensor(U0).unsqueeze(0).unsqueeze(0).cuda()
        result_U.append(torch.tensor(U0).cuda())
        u = torch.tensor(U0).cuda()
        for _ in range(m):
            u = self.model(u)
            result_U.append(u.clone())
        # print('result_U:', type(result_U))
        # print('result_U:', result_U)
        result_U = torch.cat(tuple(result_U), dim=0)

        # result_U[..., 0, :] = 1
        # result_U[..., -1, :] = 1
        # result_U[..., :, 0] = 1
        # result_U[..., :, -1] = 1
        # print('resule_U after:', result_U.shape)
        # result_U = torch.tensor(result_U)
        # print('result_U:', result_U.shape)

        # rsl = result_U.detach().cpu().numpy()
        # print('rsl:', rsl.shape)
        # for i in range(21):
        #     plt.clf()
        #     plt.imshow(rsl[i, 0, ...], cmap='hot')
        #     plt.pause(0.5)
        #     plt.ioff()
        # plt.close()

        return result_U

    def forward(self, u):
        # print('input: ', u.shape)
        fuse_u0 = False
        final_result = []
        if fuse_u0 == False:
            final_result.append(u.cuda())
        # print('u:', u.shape)
        for i in range(self.cycle):
            result = self.CG_forward(u[-1:, ...], self.single_step)  # CG_forward:  torch.Size([11, 1, 100, 100])
            # result = self.PR_forward(u[-1:, ...], self.single_step)      # PR_forward:  torch.Size([11, 1, 100, 100])
            if self.all_fusion == False:
                res_tmp = self.NN_forward(u[-1:, ...], 1)[-1:, ...]
                res = torch.zeros_like(result)
                res[-1, ...] = res_tmp
                # print('result:', result.shape)
                # print('res_tmp:', res_tmp.shape)
                # print('res:', res.shape)
                u = (1 - self.ratio) * result + self.ratio * res
                # u = (1 - self.ratio) * result + self.ratio * res
                # u = self.ratio * result + (1 - self.ratio) * res

                final_result.append(u[1:, ...])
            else:
                res = self.NN_forward(u[-1:, ...], self.single_step)
                # print('result:', result.shape)
                # print('res:', res.shape)
                u = result + self.ratio * res

                # print('res:', res.shape)
                # print('result:', result.shape)

                # u = (1 - self.ratio) * result + self.ratio * res
                # u = self.ratio * result + (1 - self.ratio) * res
                if fuse_u0 == True and i == 0:
                    final_result.append(u[:, ...])
                elif fuse_u0 == True:
                    final_result.append(u[1:, ...])
            u = u.detach()

        final_result = torch.cat(tuple(final_result), dim=0)

        # result = self.PR_forward(u)
        # res = self.NN_forward(u)
        # CG_result = self.CG_forward(u)

        # result = result + 0.0 * res

        # result = (1 - self.ratio) * result + self.ratio * (res + 1.)
        # print('ratio:', self.ratio.item())

        # self.A = torch.sigmoid(self.A) * 0.1
        # A = self.A.detach().cpu().numpy()
        # print('min:', torch.min((torch.min(self.A)), 'max:', torch.max((torch.max(self.A))))
        # result = (1.0 - self.A) * result + self.A * res

        # result = result + torch.sigmoid(self.ratio) * res
        # print('ratio:', torch.sigmoid(self.ratio).item())

        # result = result + torch.abs(self.ratio) * res
        # print('ratio:', self.ratio.item())

        # result = (1 - self.ratio) * result + self.ratio * res
        # print('ratio:', self.ratio.item())

        # result = result + 0.0 * res
        # result = 0.0 * result + res

        # result = result - self.ratio * res

        # result = CG_result + self.ratio * res

        # print('output: ', u.shape)
        return final_result

    def data_driven_forward(self, u):

        PR_list = []
        PR_list.append(u[:, ...])
        for i in range(self.intervel):
            u = self.data_driven_PR_forward(u)
            PR_list.append(u.clone().detach()[:, ...])

        PR_result = torch.cat(tuple(PR_list), dim=1)
        NN_result = self.data_driven_NN_forward(u[:, 0, ...])
        NN_result = NN_result[:, None, ...]
        # print('PR_result: ', PR_result.shape)
        # print('NN_result: ', NN_result.shape)
        res = torch.zeros_like(PR_result)
        res[:, -1:, ...] = NN_result
        # print('PR_result2: ', PR_result.shape)
        # print('res2: ', res.shape)

        result = (1 - self.ratio) * PR_result + self.ratio * res

        # print('ratio:', self.ratio.item())

        return result

    def CG_forward(self, u0, n):
        # print('in')
        u0 = u0.cuda()
        result = []
        result.append(torch.tensor(u0[0, 0, ...]))
        T = u0
        for i in range(n):
            b = T.flatten()
            # print('A:', self.A.shape)
            # print('b:', b.shape)
            T_new = self.conjugate_gradient(self.A, b)
            T = T_new.view(self.ny, self.nx)
            # print('T_new:', T.shape)
            result.append(T.clone())
        result = torch.stack(tuple(result), dim=0)
        # print('result:', result.shape)
        # for i in range(self.step):
        #     plt.clf()
        #     plt.imshow(result[i, ...], cmap='hot')
        #     plt.pause(0.5)
        #     plt.ioff()
        result = result[:, None, ...]
        # print('out')
        return result

        # 构建稀疏矩阵 A (隐式离散化)

    def build_sparse_matrix(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        Nx, Ny, alpha, dx, dy, dt, = self.nx, self.ny, self.nu, self.dx, self.dy, self.dt
        N = Nx * Ny
        r_x = alpha * dt / dx ** 2
        r_y = alpha * dt / dy ** 2

        row_indices, col_indices, values = [], [], []

        for j in range(Ny):
            for i in range(Nx):
                index = j * Nx + i
                if i == 0 or i == Nx - 1 or j == 0 or j == Ny - 1:
                    # Dirichlet 边界条件
                    row_indices.append(index)
                    col_indices.append(index)
                    values.append(1.0)
                else:
                    # 内部点离散化
                    row_indices.extend([index] * 5)
                    col_indices.extend([index, index - 1, index + 1, index - Nx, index + Nx])
                    values.extend([1 + 2 * (r_x + r_y), -r_x, -r_x, -r_y, -r_y])

        # 构造稀疏矩阵
        row_indices = torch.tensor(row_indices, dtype=torch.long, device=device)
        col_indices = torch.tensor(col_indices, dtype=torch.long, device=device)
        values = torch.tensor(values, dtype=torch.float32, device=device)

        A = torch.sparse_coo_tensor(
            torch.stack([row_indices, col_indices]), values, (N, N), device=device
        )

        return A.coalesce()  # 合并重复索引，提高计算效率

    def conjugate_gradient(self, A, b, tol=1e-6, max_iter=30):
        # print('A22')
        # print(b.device)
        # print(A.device)
        x = torch.zeros_like(b, device=b.device)
        r = b - torch.sparse.mm(A, x.unsqueeze(1)).squeeze(1)
        p = r.clone()
        rs_old = torch.dot(r, r)

        for _ in range(max_iter):
            Ap = torch.sparse.mm(A, p.unsqueeze(1)).squeeze(1)
            alpha = rs_old / torch.dot(p, Ap)
            x = x + alpha * p
            r = r - alpha * Ap
            # 会导致inplace报错
            # x += alpha * p
            # r -= alpha * Ap
            rs_new = torch.dot(r, r)

            if torch.sqrt(rs_new) < tol:
                break

            p = r + (rs_new / rs_old) * p
            rs_old = rs_new

        return x

    def get_d(self, f):

        u = f[:, 0:1, ...]

        lent = u.shape[0]
        lenx = u.shape[3]
        leny = u.shape[2]

        # # 隐式更新手段
        dt12 = Conv1dDerivative(
            DerFilter=[[[-1 / self.dt, 1 / self.dt, 0, 0, 0]]],
            deno=(1 * 1),
            kernel_size=5,
            name='partial_t')

        dt13 = Conv1dDerivative(
            DerFilter=[[[-1 / 2 * self.dt, 0, 1 / 2 * self.dt, 0, 0]]],
            deno=(1 * 1),
            kernel_size=5,
            name='partial_t')

        dt14 = Conv1dDerivative(
            DerFilter=[[[-1 / 3 * self.dt, 0, 0, 1 / 3 * self.dt, 0]]],
            deno=(1 * 1),
            kernel_size=5,
            name='partial_t')

        dt15 = Conv1dDerivative(
            DerFilter=[[[-1 / 4 * self.dt, 0, 0, 0, 1 / 4 * self.dt]]],
            deno=(1 * 1),
            kernel_size=5,
            name='partial_t')

        dt23 = Conv1dDerivative(
            DerFilter=[[[0, -1 / self.dt, 1 / self.dt, 0, 0]]],
            deno=(1 * 1),
            kernel_size=5,
            name='partial_t')

        dt24 = Conv1dDerivative(
            DerFilter=[[[0, -1 / 2 * self.dt, 0, 1 / 2 * self.dt, 0]]],
            deno=(1 * 1),
            kernel_size=5,
            name='partial_t')
        dt25 = Conv1dDerivative(
            DerFilter=[[[0, -1 / 3 * self.dt, 0, 0, 1 / 3 * self.dt]]],
            deno=(1 * 1),
            kernel_size=5,
            name='partial_t')
        dt34 = Conv1dDerivative(
            DerFilter=[[[0, 0, -1 / self.dt, 1 / self.dt, 0]]],
            deno=(1 * 1),
            kernel_size=5,
            name='partial_t')
        dt35 = Conv1dDerivative(
            DerFilter=[[[0, 0, -1 / 2 * self.dt, 0, 1 / 2 * self.dt]]],
            deno=(1 * 1),
            kernel_size=5,
            name='partial_t')
        dt45 = Conv1dDerivative(
            DerFilter=[[[0, 0, 0, -1 / self.dt, 1 / self.dt]]],
            deno=(1 * 1),
            kernel_size=5,
            name='partial_t')

        dtcn = Conv1dDerivativeCN(
            DerFilter=[[[0, -1 / self.dt, 1 / self.dt]]],
            deno=(1 * 1),
            kernel_size=3,
            name='partial_t')

        # # 显示更新手段，结果当然为0
        # dt = Conv1dDerivative(
        #     DerFilter=[[[0 , -1 / self.dt, 1 / self.dt]]],
        #     deno=(1 * 1),
        #     kernel_size=3,
        #     name='partial_t')

        # dt = Conv1dDerivative(
        #     DerFilter=[[[-1 / self.dt, 0,  1 / self.dt]]],
        #     deno=(1 * 1),
        #     kernel_size=3,
        #     name='partial_t')

        u_conv1d = u.permute(2, 3, 1, 0)  # [height(Y), width(X), c, step]
        u_conv1d = u_conv1d.reshape(lenx * leny, 1, lent)
        u_t = dtcn(u_conv1d)  # lent-2 due to no-padding
        u_t = u_t.reshape(leny, lenx, 1, lent)
        u_tcn = u_t.permute(3, 2, 0, 1)  # [step-2, c, height(Y), width(X)]

        u_conv1d = u.permute(2, 3, 1, 0)  # [height(Y), width(X), c, step]
        u_conv1d = u_conv1d.reshape(lenx * leny, 1, lent)
        u_t = dt12(u_conv1d)  # lent-2 due to no-padding
        u_t = u_t.reshape(leny, lenx, 1, lent)
        u_t12 = u_t.permute(3, 2, 0, 1)  # [step-2, c, height(Y), width(X)]

        u_conv1d = u.permute(2, 3, 1, 0)  # [height(Y), width(X), c, step]
        u_conv1d = u_conv1d.reshape(lenx * leny, 1, lent)
        u_t = dt13(u_conv1d)  # lent-2 due to no-padding
        u_t = u_t.reshape(leny, lenx, 1, lent)
        u_t13 = u_t.permute(3, 2, 0, 1)  # [step-2, c, height(Y), width(X)]

        u_conv1d = u.permute(2, 3, 1, 0)  # [height(Y), width(X), c, step]
        u_conv1d = u_conv1d.reshape(lenx * leny, 1, lent)
        u_t = dt14(u_conv1d)  # lent-2 due to no-padding
        u_t = u_t.reshape(leny, lenx, 1, lent)
        u_t14 = u_t.permute(3, 2, 0, 1)  # [step-2, c, height(Y), width(X)]

        u_conv1d = u.permute(2, 3, 1, 0)  # [height(Y), width(X), c, step]
        u_conv1d = u_conv1d.reshape(lenx * leny, 1, lent)
        u_t = dt15(u_conv1d)  # lent-2 due to no-padding
        u_t = u_t.reshape(leny, lenx, 1, lent)
        u_t15 = u_t.permute(3, 2, 0, 1)  # [step-2, c, height(Y), width(X)]

        u_conv1d = u.permute(2, 3, 1, 0)  # [height(Y), width(X), c, step]
        u_conv1d = u_conv1d.reshape(lenx * leny, 1, lent)
        u_t = dt23(u_conv1d)  # lent-2 due to no-padding
        u_t = u_t.reshape(leny, lenx, 1, lent)
        u_t23 = u_t.permute(3, 2, 0, 1)  # [step-2, c, height(Y), width(X)]

        u_conv1d = u.permute(2, 3, 1, 0)  # [height(Y), width(X), c, step]
        u_conv1d = u_conv1d.reshape(lenx * leny, 1, lent)
        u_t = dt24(u_conv1d)  # lent-2 due to no-padding
        u_t = u_t.reshape(leny, lenx, 1, lent)
        u_t24 = u_t.permute(3, 2, 0, 1)  # [step-2, c, height(Y), width(X)]

        u_conv1d = u.permute(2, 3, 1, 0)  # [height(Y), width(X), c, step]
        u_conv1d = u_conv1d.reshape(lenx * leny, 1, lent)
        u_t = dt25(u_conv1d)  # lent-2 due to no-padding
        u_t = u_t.reshape(leny, lenx, 1, lent)
        u_t25 = u_t.permute(3, 2, 0, 1)  # [step-2, c, height(Y), width(X)]

        u_conv1d = u.permute(2, 3, 1, 0)  # [height(Y), width(X), c, step]
        u_conv1d = u_conv1d.reshape(lenx * leny, 1, lent)
        u_t = dt34(u_conv1d)  # lent-2 due to no-padding
        u_t = u_t.reshape(leny, lenx, 1, lent)
        u_t34 = u_t.permute(3, 2, 0, 1)  # [step-2, c, height(Y), width(X)]

        u_conv1d = u.permute(2, 3, 1, 0)  # [height(Y), width(X), c, step]
        u_conv1d = u_conv1d.reshape(lenx * leny, 1, lent)
        u_t = dt35(u_conv1d)  # lent-2 due to no-padding
        u_t = u_t.reshape(leny, lenx, 1, lent)
        u_t35 = u_t.permute(3, 2, 0, 1)  # [step-2, c, height(Y), width(X)]

        u_conv1d = u.permute(2, 3, 1, 0)  # [height(Y), width(X), c, step]
        u_conv1d = u_conv1d.reshape(lenx * leny, 1, lent)
        u_t = dt45(u_conv1d)  # lent-2 due to no-padding
        u_t = u_t.reshape(leny, lenx, 1, lent)
        u_t45 = u_t.permute(3, 2, 0, 1)  # [step-2, c, height(Y), width(X)]

        self.lap_2d_op = [[[
            [0, 1 / self.dd ** 2, 0],
            [1 / self.dd ** 2, -4 / self.dd ** 2, 1 / self.dd ** 2],
            [0, 1 / self.dd ** 2, 0]
        ]]]

        self.laplace = Conv2dDerivative(
            DerFilter=self.lap_2d_op,
            deno=(1),
            kernel_size=3,
            name='laplace_operator')

        laplace_u = self.laplace(u)  # 201x1x128x128

        # print('loss u:', u)
        # print('loss u_t:', u_t)
        # print('loss u_lap:', laplace_u)
        # input('000')

        return u, u_tcn, u_t12, u_t13, u_t14, u_t15, u_t23, u_t24, u_t25, u_t34, u_t35, u_t45, laplace_u

    def get_phy_loss(self, U):

        u, u_tcn, u_t12, u_t13, u_t14, u_t15, u_t23, u_t24, u_t25, u_t34, u_t35, u_t45, u_lap = self.get_d(U)

        # f = - self.nu * u_lap

        fcn = u_tcn[:-1, ...] - (self.nu / 2 * u_lap[:-1, ...]) - (self.nu / 2 * u_lap[1:, ...])

        f12 = u_t12 - self.nu * u_lap  # f: torch.Size([21, 1, 100, 100])
        f13 = u_t13 - self.nu * u_lap
        f14 = u_t14 - self.nu * u_lap
        f15 = u_t15 - self.nu * u_lap  # f: torch.Size([21, 1, 100, 100])
        f23 = u_t23 - self.nu * u_lap
        f24 = u_t24 - self.nu * u_lap
        f25 = u_t25 - self.nu * u_lap  # f: torch.Size([21, 1, 100, 100])
        f34 = u_t34 - self.nu * u_lap
        f35 = u_t35 - self.nu * u_lap
        f45 = u_t45 - self.nu * u_lap
        # f = self.nu * u_lap - u_t

        # f_imp = f.detach().cpu().numpy()
        # for i in range(21):
        #     plt.clf()
        #     plt.imshow(f_imp[i, 0, 1:-1, 1:-1], cmap='hot')
        #     # plt.title('f importance')
        #     plt.colorbar()
        #     plt.pause(0.5)
        #     plt.ioff()
        # plt.close()

        # print('pass')

        return fcn, f12, f13, f14, f15, f23, f24, f25, f34, f35, f45

    def get_d_data(self, f):

        u = f[:, :, 0:1, ...]

        # b, t, c, h, w

        lenb = u.shape[0]
        lent = u.shape[1]
        lenx = u.shape[4]
        leny = u.shape[3]

        dtcn = Conv1dDerivativeCN(
            DerFilter=[[[0, -1 / self.dt, 1 / self.dt]]],
            deno=(1 * 1),
            kernel_size=3,
            name='partial_t')

        u_conv1d = u.permute(0, 3, 4, 2, 1)  # [height(Y), width(X), c, step]   # b, t, c, h, w -> b, h, w, c, t
        u_conv1d = u_conv1d.reshape(lenb * lenx * leny, 1, lent)  # b, h, w, c, t _> bhw, c, t
        u_t = dtcn(u_conv1d)  # lent-2 due to no-padding
        u_t = u_t.reshape(lenb, leny, lenx, 1, lent)  # bhw, c, t -> b, h, w, c, t
        u_tcn = u_t.permute(0, 4, 3, 1,
                            2)  # [step-2, c, height(Y), width(X)]           # b, h, w, c, t -> b, t, c, h, w

        # u_conv1d = u.permute(2, 3, 1, 0)  # [height(Y), width(X), c, step]
        # u_conv1d = u_conv1d.reshape(lenx * leny, 1, lent)
        # u_t = dtcn(u_conv1d)  # lent-2 due to no-padding
        # u_t = u_t.reshape(leny, lenx, 1, lent)
        # u_tcn = u_t.permute(3, 2, 0, 1)  # [step-2, c, height(Y), width(X)]

        self.lap_2d_op = [[[
            [0, 1 / self.dd ** 2, 0],
            [1 / self.dd ** 2, -4 / self.dd ** 2, 1 / self.dd ** 2],
            [0, 1 / self.dd ** 2, 0]
        ]]]

        self.laplace = Conv2dDerivative(
            DerFilter=self.lap_2d_op,
            deno=(1),
            kernel_size=3,
            name='laplace_operator')

        # u_conv2d = u.permute(0, 3, 4, 2, 1)  # [height(Y), width(X), c, step]   # b, t, c, h, w -> b, t,
        u_conv2d = u.reshape(lenb * lent, 1, lenx, leny)  # b, h, w, c, t _> bhw, c, t
        laplace_u = self.laplace(u_conv2d)
        u_lap = laplace_u.reshape(lenb, lent, 1, leny, lenx)  # bhw, c, t -> b, h, w, c, t

        # print('u: ', u.shape)
        # print('u_tcn: ', u_tcn.shape)
        # print('u_lap: ', u_lap.shape)

        return u, u_tcn, u_lap

    def get_phy_loss_data(self, U):

        u, u_tcn, u_lap = self.get_d_data(U)

        u, u_tcn, u_lap = u[..., 1:-1, 1:-1], u_tcn[..., 1:-1, 1:-1], u_lap[..., 1:-1, 1:-1]

        # comp = (u[..., 0:-2, 1:-1] + u[..., 2:, 1:-1] + u[..., 1:-1, 0:-2] + u[..., 1:-1, 2:] - 4*u[..., 1:-1, 1:-1]) / 1e-4
        #
        #
        #
        # print(comp[0, 1, 0, ...])
        #
        # print(u_lap[0, 1, 0, 1:-1, 1:-1])
        # print(comp.shape)
        # print(u_lap.shape)

        # print('u:, ', u.shape)
        # print('u_tcn:, ', u_tcn.shape)
        # print('u_lap:, ', u_lap.shape)

        # print(u_tcn.sum())
        # print(u_lap.sum())

        # print(u_tcn[0, 0:2, ...])

        fcn = u_tcn[:, :-1, ...] - (self.nu / 2 * u_lap[:, :-1, ...]) - (self.nu / 2 * u_lap[:, 1:, ...])

        # print(fcn[0, 0:2, ...])
        #
        # for i in range(5):
        #     plt.clf()
        #     plt.imshow(fcn[0, i, 0, ...].detach().cpu().numpy(), cmap='bwr', origin='lower')
        #     plt.colorbar()
        #     plt.pause(2)
        #     plt.ioff()

        return fcn

    def train_nn(self):

        for i in range(6060):
            # u0 = self.get_u0()
            # print('u0:', u0.shape)
            # plt.imshow(u0, cmap='hot')
            # plt.show()
            u0 = torch.tensor(self.get_u0(), dtype=torch.float32).unsqueeze(0).unsqueeze(0).cuda()
            result = self.forward(u0)  # result.shape: (41, 1, 100, 100)
            if i % self.save_step == 0:
                long_result = []
                long_result.append(result.detach().cpu().numpy())
                for j in range(int(10000 / self.step)):
                    # print('next:', long_result[-1][-1, ...][None, ...].shape)
                    next_u0 = torch.from_numpy(long_result[-1][-1, ...][None, ...])
                    # print(next_u0.shape)
                    tmp_result = self.forward(next_u0)
                    long_result.append(tmp_result[1:, ...].detach().cpu().numpy())
                long_result = np.concatenate(long_result, axis=0)
                # for i in range(1001):
                #     plt.clf()
                #     plt.imshow(long_result[i, 0, ...], cmap='hot')
                #     plt.title('f importance' + str(i))
                #     plt.colorbar()
                #     plt.pause(0.001)
                #     plt.ioff()
                # print('long_result: ', long_result.shape)   # (1041, 1, 100, 100)

            fcnr, f12r, f13r, f14r, f15r, f23r, f24r, f25r, f34r, f35r, f45r = self.get_phy_loss(result)
            # f = f_r[1:, ..., 1:-1, 1:-1]
            fcnr, f12r, f13r, f14r, f15r, f23r, f24r, f25r, f34r, f35r, f45r = \
                fcnr, f12r[2:, ...], f13r[2:, ...], f14r[2:-1, ...], f15r[2:-2, ...], \
                    f23r[1:, ...], f24r[1:-1, ...], f25r[1:-2, ...], f34r[:-1, ...], f35r[:-2, ...], f45r[:-2, ...]

            fcnr = torch.abs(fcnr)
            # for k in range(self.step - 1):
            #     f[k:k+1, ...] = f[k:k+1, ...] * (1 - math.exp(-0.3*k))

            # for k in range(self.step - 1):
            #     f[k:k+1, ...] = f[k:k+1, ...] * (math.exp(-0.3*k))

            # f_imp = f.detach().cpu().numpy()
            # for i in range(21):
            #     plt.clf()
            #     plt.imshow(f_imp[i, 0, ...], cmap='hot')
            #     plt.title('f importance')
            #     plt.colorbar()
            #     plt.pause(2)
            #     plt.ioff()
            # plt.close()

            fcn = self.loss_fn(fcnr, torch.zeros_like(fcnr))
            f12 = self.loss_fn(f12r, torch.zeros_like(f12r))
            f13 = self.loss_fn(f13r, torch.zeros_like(f13r))
            f14 = self.loss_fn(f14r, torch.zeros_like(f14r))
            f15 = self.loss_fn(f15r, torch.zeros_like(f15r))
            f23 = self.loss_fn(f23r, torch.zeros_like(f23r))
            f24 = self.loss_fn(f24r, torch.zeros_like(f24r))
            f25 = self.loss_fn(f25r, torch.zeros_like(f25r))
            f34 = self.loss_fn(f34r, torch.zeros_like(f34r))
            f35 = self.loss_fn(f35r, torch.zeros_like(f35r))
            f45 = self.loss_fn(f45r, torch.zeros_like(f45r))

            # loss = f12 + f13 + f14 + f15 + f23 + f24 + f25 + f34 + f35 + f45
            # loss = fcn

            loss = fcn  # + f34

            if i == 0:
                self.mse0_save(result)
                self.long_mse0_save(long_result)
                print('ratio init:', self.ratio.item())
                print('MSE0 init:', self.MSE0)
                print('MSE0 frame init:', self.frameMSE)
                print('init finished, ratio is ', self.ratio.item())
                # self.ratio = torch.nn.Parameter(torch.tensor(1e-15, requires_grad=True))

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            # self.scheduler.step()

            if i % self.print_step == 0:
                print('ratio:', self.ratio.item())
                # print('epoch:', i, ', loss:', loss.item(), ', f12:', f12.item(), ', f13:', f13.item(), ', f14:', f14.item(),
                # ', f15:', f15.item(),
                # ', f23:', f23.item(),
                # ', f24:', f24.item(),
                # ', f25:', f25.item(),
                # ', f34:', f34.item(),
                # ', f35:', f35.item(),
                # ', f45:', f45.item()
                # )
                print('[epoch]:', i, ', [seed]:', seed, ', [loss]:', loss.item(), ', [fcn]:', fcn.item())
                self.comp_gt_only(result)
                # self.plot_u0()

            if i % self.save_step == 0:
                self.print_frame_relative_mse(result)

            if i % self.save_step == 0:
                os.mkdir('output/result/' + str(i))
                os.mkdir('output/result/' + str(i) + '/loss')
                os.mkdir('output/result/' + str(i) + '/MSE')
                os.mkdir('output/result/' + str(i) + '/longMSE')
                self.current_epoch = str(i) + '/'

                self.comp_with_gt(result)
                self.comp_with_long_gt(long_result)
                # self.plot_gt()
                self.plot_result(result)
                self.plot_diff(result)
                # for j in range(rsl.shape[0]):
                #     plt.clf()
                #     plt.imshow(rsl[j, 0, ...], cmap='hot', vmin=1., vmax=2.)
                #     plt.colorbar()
                #     plt.pause(0.1)
                #     plt.ioff()
                # plt.close()

        #     # print('result:', result)
        #     # self.result_vector.append([phy_loss.item(), f.item(), g.item(), c.item()])
        #
        #     self.comp_with_gt(result)
        #
        #     if i % 50 == 0 and i != 0:
        #         os.mkdir('output/result/' + str(i))
        #         os.mkdir('output/result/' + str(i) + '/loss')
        #         os.mkdir('output/result/' + str(i) + '/MSE')
        #         self.current_epoch = str(i) + '/'
        #         self.print_phylosses()
        #         self.print_mse_his()
        #         self.evaluate(result)
        #         self.plot_u_diff(result)
        #         # input('input......')
        #         self.plot_vector(result)
        #         self.plot_stream(result)
        #         self.plot_nr(f_r)
        #         # input('请输入回车以继续......')
        #
        #     # if i % 50 == 0:
        #     #     for j in range(result.shape[0]):
        #     #         plt.clf()
        #     #         plt.imshow(result[j, 0, ...].detach().numpy(), cmap='bwr')
        #     #         plt.pause(0.1)
        #     #         plt.ioff()
        #     #     plt.close()
        #
        # self.result_vector = np.array(self.result_vector)
        # input("请输入任意字符......")
        # self.plot_vector(result)
        # self.plot_stream(result)
        # print('result_vector:', self.result_vector.shape)
        #
        # x = range(len(self.result_vector[:, 0]))
        # plt.plot(x, self.result_vector[:, 0], marker='o', linestyle='-', label='phyloss')
        # plt.plot(x, self.result_vector[:, 1], marker='o', linestyle='--', label='f')
        # # plt.plot(x, self.result_vector[:, 2], marker='o', linestyle='-.', label='g')
        # plt.plot(x, self.result_vector[:, 3], marker='o', linestyle=':', label='c')
        # plt.show()

        # gt = np.load('cross_result.npy')
        # print()

    def print_frame_relative_mse(self, result):
        result = result.detach().cpu().numpy()
        mse_list = np.zeros(self.step + 1)
        for i in range(self.step + 1):
            mse = mean_squared_error(self.gt[i, ...].flatten(), result[i, ...].flatten())
            mse_list[i] = mse
        frame_mse_relative = mse_list[:self.step + 1] - self.frameMSE
        # print('relative frame mse:', frame_mse_relative[:self.step + 1])

    def save_result(self, result):
        result = result.detach().cpu().numpy()
        np.save('output/result/' + self.current_epoch + 'result.npy')

    def mse0_save(self, result):
        result = result.detach().cpu().numpy()

        MSE = mean_squared_error(self.gt.flatten(), result.flatten())
        MSEcutu00 = mean_squared_error(self.gt[1:, ...].flatten(), result[1:, ...].flatten())

        self.MSE0 = MSE
        self.MSEcutu00 = MSEcutu00

        # print('#:' , self.gt[1:, ...].shape)
        mse_list = np.zeros(self.step + 1)
        for i in range(self.step + 1):
            mse = mean_squared_error(self.gt[i, ...].flatten(), result[i, ...].flatten())
            mse_list[i] = mse

        self.frameMSE = mse_list[:self.step + 1]

    def long_mse0_save(self, long_result):
        long_result = long_result  # .detach().cpu().numpy()

        long_MSE = mean_squared_error(self.long_gt[:self.long_step].flatten(),
                                      long_result[:self.long_step].flatten())  # 1-1000 1001个
        long_MSE0cutu0 = mean_squared_error(self.long_gt[1:self.long_step, ...].flatten(),
                                            long_result[1:self.long_step, ...].flatten())

        self.long_MSE0 = long_MSE
        self.long_MSE0cutu0 = long_MSE0cutu0

        # print('#:' , self.gt[1:, ...].shape)
        mse_list = np.zeros(self.long_step + 1)
        for i in range(self.long_step + 1):
            mse = mean_squared_error(self.long_gt[i, ...].flatten(), long_result[i, ...].flatten())
            mse_list[i] = mse

        self.long_frameMSE = mse_list[:self.long_step + 1]

        # print('!!!!!!!!!!:', self.long_frameMSE)
        # os.system('pause')

    def comp_with_long_gt(self, long_result):

        # np.save('output/result/' + self.current_epoch + 'longMSE/long_result.npy', long_result)

        long_gt = self.long_gt[:self.long_step + 1]
        long_result = long_result[:self.long_step + 1]
        print('long_result.shape:', long_result.shape)
        print('long_gt.shape:', self.long_gt.shape)

        MSE = mean_squared_error(long_gt.flatten(), long_result.flatten())
        print('long MSE:', MSE)
        np.save('output/result/' + self.current_epoch + 'longMSE/total_long_MSE.npy', MSE)

        mse_list = np.zeros(self.long_step + 1)
        for i in range(self.long_step + 1):
            mse = mean_squared_error(long_gt[i, ...].flatten(), long_result[i, ...].flatten())
            mse_list[i] = mse
        # np.save('output/result/' + self.current_epoch + 'longMSE/long_result.npy', long_result)
        np.save('output/result/' + self.current_epoch + 'longMSE/long_frame_MSE.npy', mse_list[:self.long_step + 1])

        plt.plot(range(self.long_step + 1), mse_list[:self.long_step + 1])
        plt.title('long MSE (result and gt) with frame')
        plt.savefig('output/result/' + self.current_epoch + 'longMSE/long_MSE.png')
        plt.close()

        # print('self.frameMSE:', self.frameMSE.shape)
        plt.plot(range(1, self.long_step + 1), self.long_frameMSE[1:self.long_step + 1], color='blue',
                 label='prior MSE')
        plt.plot(range(1, self.long_step + 1), mse_list[1:self.long_step + 1], color='red', label='our MSE')
        # plt.plot(range(1, self.long_step + 1), self.long_percnn_mse[1:self.long_step + 1], color='black', label='percnn MSE')
        plt.title('long MSE (result and gt) with frame cut u0')
        plt.legend()
        plt.savefig('output/result/' + self.current_epoch + 'longMSE/long_MSEcutu0_prior.png')
        plt.close()

        # plt.plot(range(1, self.long_step + 1), self.long_frameMSE[1:self.long_step + 1], color='blue',
        #          label='prior MSE')
        plt.plot(range(1, self.long_step + 1), mse_list[1:self.long_step + 1], color='red', label='our MSE')
        # plt.plot(range(1, self.long_step + 1), self.long_percnn_mse[1:self.long_step + 1], color='black', label='percnn MSE')
        plt.title('long MSE (result and gt) with frame cut u0')
        plt.legend()
        plt.savefig('output/result/' + self.current_epoch + 'longMSE/long_MSEcutu0_percnn.png')
        plt.close()

        plt.plot(range(1, self.long_step + 1), np.sqrt(self.long_frameMSE[1:self.long_step + 1]), color='blue',
                 label='prior MSE')
        plt.plot(range(1, self.long_step + 1), np.sqrt(mse_list[1:self.long_step + 1]), color='red', label='our MSE')
        # plt.plot(range(1, self.long_step + 1), self.long_percnn_mse[1:self.long_step + 1], color='black', label='percnn MSE')
        plt.title('long MSE (result and gt) with frame cut u0')
        plt.legend()
        plt.savefig('output/result/' + self.current_epoch + 'longMSE/long_MSEcutu0_tri.png')
        plt.close()

        MSE = mean_squared_error(self.long_gt.flatten(), long_result.flatten())

        print('long MSE:', MSE)

        MSEcutu0 = mean_squared_error(self.long_gt[1:, ...].flatten(), long_result[1:, ...].flatten())
        mse_relativecutu0 = MSEcutu0 - self.long_MSE0cutu0
        print('relative long mse(no u0):', mse_relativecutu0)

    def comp_with_gt(self, result):
        result = result.detach().cpu().numpy()
        # print('result.shape:', result.shape)
        # print('gt.shape:', self.gt.shape)
        MSE = mean_squared_error(self.gt.flatten(), result.flatten())

        # print('total MSE:', MSE)

        np.save('output/result/' + self.current_epoch + 'total_MSE.npy', MSE)

        mse_list = np.zeros(self.step + 1)
        for i in range(self.step + 1):
            mse = mean_squared_error(self.gt[i, ...].flatten(), result[i, ...].flatten())
            mse_list[i] = mse

        np.save('output/result/' + self.current_epoch + 'result.npy', result)

        print('frame MSE:', mse_list)

        np.save('output/result/' + self.current_epoch + 'frame_MSE.npy', mse_list[:self.step + 1])

        plt.plot(range(self.step + 1), np.sqrt(mse_list[:self.step + 1]))
        plt.title('MSE (result and gt) with frame')
        plt.savefig('output/result/' + self.current_epoch + 'MSE.png')
        plt.close()

        # print('self.frameMSE:', self.frameMSE.shape)
        plt.plot(range(self.step), np.sqrt(self.frameMSE[1:self.step + 1]), color='blue', label='prior MSE')
        plt.plot(range(self.step), np.sqrt(mse_list[1:self.step + 1]), color='red', label='our MSE')
        plt.title('MSE (result and gt) with frame cut u0')
        plt.legend()
        plt.savefig('output/result/' + self.current_epoch + 'MSEcutu0.png')
        plt.close()

        mse_relative = MSE - self.MSE0
        print('relative mse:', mse_relative)
        np.save('output/result/' + self.current_epoch + 'total_MSE_relative_MSE0.npy', MSE)
        frame_mse_relative = mse_list[:self.step + 1] - self.frameMSE
        print('relative frame mse:', frame_mse_relative[:self.step + 1])
        plt.plot(range(self.step + 1), frame_mse_relative[:self.step + 1])
        plt.title('relative MSE with frame')
        plt.savefig('output/result/' + self.current_epoch + 'frame_MSE_relative_MSE0.png')
        plt.close()

        MSEcutu0 = mean_squared_error(self.gt[1:, ...].flatten(), result[1:, ...].flatten())
        mse_relativecutu0 = MSEcutu0 - self.MSEcutu00
        print('relative mse(no u0):', mse_relativecutu0)
        np.save('output/result/' + self.current_epoch + 'total_MSE_relative_MSE0_no_u0.npy', MSEcutu0)
        frame_mse_relative_no_u0 = mse_list[1:self.step + 1] - self.frameMSE[1:]
        # print('relative frame mse(no u0):', frame_mse_relative[:self.step + 1])
        plt.plot(range(self.step + 1 - 1), frame_mse_relative[1:self.step + 1])
        plt.title('relative MSE with frame(no u0)')
        plt.savefig('output/result/' + self.current_epoch + 'frame_MSE_relative_MSE0_cut_u0.png')
        plt.close()

    def print_diff_absolute(self, result):
        rsl = np.array(result.clone())
        di = np.abs(self.gt - rsl)
        vmin = np.min(di.flatten())
        vmax = np.max(di.flatten())

        for i in range(self.step + 1):
            diff = result[i, ...] - self.gt[i, ...]
            plt.clf()
            plt.imshow(diff, cmap='hot', vmin=vmin, vmax=vmax)
            plt.colorbar()
            # plt.show()
            # plt.show()
            # if i == 0:
            #     input('input any button.')
            plt.pause(0.01)
            plt.ioff()
        plt.close()

    def print_diff_relative(self, result):
        for i in range(self.step + 1):
            diff = result[i, ...] - self.gt[i, ...]
            plt.clf()
            plt.imshow(diff, cmap='hot')
            plt.colorbar()
            # plt.show()
            # plt.show()
            # if i == 0:
            #     input('input any button.')
            plt.pause(0.01)
            plt.ioff()
        plt.close()

    def plot_diff(self, result):
        result = result.detach().cpu().numpy()
        for i in range(self.step + 1):
            plt.clf()
            # plt.imshow(np.abs(self.gt[i, ...] - result[i, 0, ...]) / np.abs(self.gt[i, ...]), cmap='hot')
            plt.imshow(np.abs(self.gt[i, ...] - result[i, 0, ...]), cmap='hot')
            plt.colorbar()
            plt.title('plot diff')
            plt.savefig('output/cache/diff/' + str(i) + '.png')
            # plt.pause(0.5)
            # plt.ioff()
        plt.close()
        self.mp4('output/cache/diff', 'output/result/' + self.current_epoch + 'diff.mp4')

    def plot_gt(self):

        for i in range(self.step + 1):
            plt.clf()
            plt.imshow(np.abs(self.gt[i, ...]), cmap='hot')
            plt.colorbar()
            plt.title('plot gt')
            plt.pause(0.5)
            plt.ioff()
        plt.close()

    def plot_result(self, result):
        result = result.detach().cpu().numpy()
        for i in range(self.step + 1):
            plt.clf()
            plt.imshow(np.abs(result[i, 0, ...]), cmap='hot', vmin=0, vmax=1)
            plt.colorbar()
            plt.title('plot result')
            plt.savefig('output/cache/result/' + str(i) + '.png')
            # plt.pause(0.1)
            # plt.ioff()
        plt.close()
        self.mp4('output/cache/result', 'output/result/' + self.current_epoch + 'result.mp4')

    def plot_u0(self):
        plt.imshow(self.u0, cmap='hot')
        plt.show()

    def get_u0(self):
        return self.u0.copy()

    def lap(self, f):

        k_lap = np.array(
            [[0., 1., 0.],
             [1., -4., 1.],
             [0., 1., 0.]]
        )

        f = f.clone().detach().unsqueeze(0).unsqueeze(0)
        # k_lap = torch.tensor(k_lap, dtype=torch.float32).unsqueeze(0).unsqueeze(0)#.cuda()
        k_lap = torch.tensor(k_lap, dtype=torch.float32).unsqueeze(0).unsqueeze(0).cuda()
        diff = F.conv2d(f, k_lap, padding=1)  # / (self.dd**2)
        diff = diff.squeeze()  # .detach().numpy()
        diff[0, :] = diff[:, 0] = diff[-1, :] = diff[:, -1] = 0.0

        return diff

    def data_driven_lap(self, f):
        # print(f.shape)
        b, t, c, h, w = f.shape

        k_lap = torch.tensor([
            [0., 1., 0.],
            [1., -4., 1.],
            [0., 1., 0.]
        ], dtype=torch.float32).view(1, 1, 3, 3).cuda()  # 形状调整为 [1, 1, 3, 3]

        f = f.reshape(b * t, c, h, w)  # b, h, w, c, t _> bhw, c, t

        diff = F.conv2d(
            input=f,
            weight=k_lap,
            padding=1
        )

        diff = diff.reshape(b, t, c, h, w)  # bhw, c, t -> b, h, w, c, t

        diff[..., 0, :] = diff[..., :, 0] = diff[..., -1, :] = diff[..., :, -1] = 0.0
        return diff

    def init_weights_normal(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, mean=0.0, std=1e-6)
            nn.init.zeros_(m.bias)

    def calculate_padding(self, kernel_size):
        pad = ((kernel_size - 1) * self.stride + 1 - self.stride) // 2
        return pad

    def comp_gt_only(self, result):
        result = result.clone().detach().cpu().numpy()
        MSE = mean_squared_error(self.gt.flatten(), result.flatten())
        mse_relative = MSE - self.MSE0
        MSEcutu0 = mean_squared_error(self.gt[1:, ...].flatten(), result[1:, ...].flatten())
        mse_relativecutu0 = MSEcutu0 - self.MSEcutu00
        print('[total RMSE]:', np.sqrt(MSE), '[relative MSE]:', mse_relative, '[RMSE(no u0)]:', np.sqrt(MSEcutu0),
              '[relative MSE(no u0)]:',
              mse_relativecutu0)

    def mp4(self, inpath, outpath):

        input_folder = inpath
        output_video = outpath

        # 获取文件夹中的所有 PNG 文件并按顺序排序
        files = os.listdir(input_folder)
        files = sorted(files, key=lambda x: int(re.findall(r'\d+', x)[0]))

        # 获取第一个 PNG 文件的尺寸，用于设置视频输出
        img = cv2.imread(os.path.join(input_folder, files[0]))
        height, width, _ = img.shape

        # 设置输出视频的帧率和编码器
        fps = 2  # 帧率
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 MP4V 编码器

        # 创建 VideoWriter 对象
        out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

        # 逐个将 PNG 文件写入视频
        for file in files:
            img = cv2.imread(os.path.join(input_folder, file))
            out.write(img)

        # 释放 VideoWriter 对象
        out.release()

        print(f'视频已保存为 {output_video}')

        def delete_folder_contents(folder_path):
            # 确保文件夹路径存在
            if not os.path.exists(folder_path):
                print(f"文件夹 '{folder_path}' 不存在。")
                return

            # 确保路径是一个文件夹
            if not os.path.isdir(folder_path):
                print(f"'{folder_path}' 不是一个有效的文件夹路径。")
                return

            # 递归删除文件夹内容
            for filename in os.listdir(folder_path):
                file_path = os.path.join(folder_path, filename)
                try:
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(f"删除 '{file_path}' 时出错: {e}")

            print(f"文件夹 '{folder_path}' 中的所有文件和文件夹已删除。")

        # 使用示例
        delete_folder_contents(input_folder)


class TimeSeriesMatrixDataset(Dataset):
    def __init__(self, data_tensor):
        """
        Args:
            data_tensor: 形状为 (T, H, W) 的张量，T 是时间步，H 和 W 是矩阵的高度和宽度。
        """
        self.data = data_tensor[:, None, ...]  # 形状: (1000, 100, 100)
        self.T = data_tensor.shape[0]  # 时间步数

    def __len__(self):
        # 最后一个样本是 (T-2, T-1)，因此总长度是 T-1
        return self.T - interval

    def __getitem__(self, idx):
        """
        返回第 idx 和 idx+1 的矩阵对 (input, target)。
        """
        if idx >= self.T - interval:
            raise IndexError("Index out of range. Dataset length is T-1.")

        # 获取当前矩阵 (idx) 和下一个矩阵 (idx+1)
        input_matrix = self.data[idx]  # 形状: (100, 100)
        target_matrix = self.data[idx + interval]  # 形状: (100, 100)

        return input_matrix, target_matrix


def rollout_data_driven(net, steps):
    """Rollout net.data_driven_forward for a fixed number of steps, returning [T,H,W]."""
    u0 = net.get_u0()
    u = torch.tensor(u0[None, None, None, ...], dtype=torch.float32).cuda()
    result = [u0.copy()]
    for _ in range(steps):
        u = net.data_driven_forward(u[:, -1:, ...])
        result.append(u[0, -1, 0, ...].clone().detach().cpu().numpy())
    return np.array(result)


def export_web_bundle(net, epoch, short_prediction, short_gt):
    long_steps = int(getattr(net, "long_step", max(len(short_gt) - 1, 0)))
    long_prediction = rollout_data_driven(net, long_steps)
    long_gt = net.getgtsympy(long_steps + 1, delt=net.dt * interval)

    payload = ExportPayload(
        prediction_short=short_prediction,
        gt_short=short_gt,
        prediction_long=long_prediction,
        gt_long=long_gt,
        model_name="pinn",
        epoch=int(epoch),
        dt=float(net.dt * interval),
        dx=float(net.dx),
        dy=float(net.dy),
        extra_meta={
            "interval": int(interval),
            "nu": float(net.nu),
            "source": "Hea0.1.19.py/test",
        },
    )
    out_dir = export_prediction_bundle('.', payload)
    print(f"[web export] exported to: {out_dir}")


def test(net, fin=False, epo=0):
    global mse0
    # test phase
    test_len = 100 // interval
    test_gt = net.getgtsympy(test_len + 1, delt=1e-5 * interval)
    u = test_gt[None, 0:1, None, ...]
    # print('uuu: ', u.shape)
    result = []
    result.append(u[0, 0, ...])
    u = torch.tensor(u, dtype=torch.float32).cuda()
    for _ in tqdm(range(test_len)):
        u = net.data_driven_forward(u[:, -1:, ...])
        # print('u2: ', u.shape)
        result.append(u[0, -1, ...].clone().detach().cpu().numpy())
    result = np.array(result)
    # print(result.shape)
    np.save('data/output/result/result_' + str(epo) + '.npy', result)
    export_web_bundle(net=net, epoch=epo, short_prediction=result, short_gt=test_gt)
    RMSE = []
    for i in trange(test_len):
        rmse = np.sqrt(mean_squared_error(test_gt[i, ...].flatten(), result[i, ...].flatten()))
        RMSE.append(rmse)
    # if fin == False:
    #    mse0 = RMSE

    if fin == True:
        plt.plot(mse0, label='origin')
        plt.plot(RMSE, label=str(epo))
        plt.legend()
        plt.savefig('data/output/result/fin_RMSE_fig' + str(epo) + '.png')
        # plt.legend()
        # plt.show()
    else:
        plt.plot(RMSE, label='start')
        # plt.savefig('data/output/result/start_RMSE_fig.png')
        plt.xlabel('time')
        plt.ylabel('RMSE')
        plt.legend()
        plt.savefig('data/output/result/start_RMSE_fig.png')
        mse0 = RMSE
    plt.close()


if __name__ == '__main__':
    # prnet = PRNet()
    # u0 = prnet.get_u0()
    # result = prnet.PR_forward(u0)
    # prnet.comp_with_gt(result)
    # prnet.print_diff_absolute(result)
    # prnet.print_diff_relative(result)

    seed = np.random.randint(1, 100000)
    seed = 50976
    set_random_seed(mySeed=seed)  # mySeed = 4,95.5左右
    print('using seed:', seed)


    def delete_folder_contents(folder_path):
        # 确保文件夹路径存在
        if not os.path.exists(folder_path):
            print(f"文件夹 '{folder_path}' 不存在。")
            return

        # 确保路径是一个文件夹
        if not os.path.isdir(folder_path):
            print(f"'{folder_path}' 不是一个有效的文件夹路径。")
            return

        # 递归删除文件夹内容
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"删除 '{file_path}' 时出错: {e}")

        print(f"文件夹 '{folder_path}' 中的所有文件和文件夹已删除。")


    delete_folder = './data/output/result/'
    # 使用示例
    delete_folder_contents(delete_folder)
    prnet = PRNet().cuda()
    # prnet.train_nn()

    tensor = torch.randn(64, 1, 1, 101, 101).to(torch.float32).cuda()
    output = prnet.data_driven_forward(tensor)
    print(output.shape)

    u = torch.tensor(prnet.get_u0()[None, None], dtype=torch.float32).cuda()
    # print(u.shape)
    # result = []
    # for i in range(1000):
    #     u = prnet.data_driven_forward(u)
    #     result.append(u.clone().detach().cpu().numpy())
    # result = np.array(result)
    # print(result.shape)
    #
    # for j in range(1000):
    #     plt.clf()
    #     plt.imshow(result[j][0, 0, ...], cmap='hot', origin='lower')
    #     plt.colorbar()
    #     plt.pause(0.002)
    #     plt.ioff()

    gt = prnet.getgtsympy(20 // interval, 1e-5 * interval)
    print('gt: ', gt.shape)

    # 创建数据集
    dataset = TimeSeriesMatrixDataset(gt)

    # 创建 DataLoader，设置批次大小为64
    dataloader = DataLoader(
        dataset,
        batch_size=16 // interval,  # 每次加载64个样本
        shuffle=True,  # 是否打乱数据
        num_workers=0,  # 多进程加载（可选）
        drop_last=True  # 丢弃最后不足一个批次的数据
    )
    loss_fn = prnet.loss_fn
    optimizer = torch.optim.Adam(prnet.parameters(), lr=1e-5)
    with torch.no_grad():
        test(prnet)

    epo = 1000100
    for epoch in range(epo):
        total_loss = 0.
        tp_loss = 0.
        td_loss = 0.
        for batch_idx, (batch_input, batch_target) in enumerate(dataloader):
            batch_input, batch_target = batch_input[:, None, ...].to(torch.float32).cuda(), batch_target[:, None,
                                                                                            ...].to(
                torch.float32).cuda()
            # print('batch_input: ', batch_input.shape)
            # print('batch_target: ', batch_target.shape)
            optimizer.zero_grad()
            pred = prnet.data_driven_forward(batch_input)
            # print('pred: ', pred.shape)

            phy_loss = prnet.get_phy_loss_data(pred)
            phy_loss = loss_fn(phy_loss, torch.zeros_like(phy_loss))
            data_loss = loss_fn(pred[:, -1:, ...], batch_target)
            # print('phy_loss: ', phy_loss.shape)
            # print('data_loss: ', data_loss.shape)
            # print(data_loss)

            #loss = 1e3 * phy_loss + 1e9 * data_loss
            loss = 1.0 * phy_loss + 0.0 * data_loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            tp_loss += phy_loss.item()
            td_loss += data_loss.item()

        print(
            f"\r seed: {seed} interval: {interval} It: {epoch} Loss: {total_loss:.5e} phy_loss: {tp_loss:.5e} data_loss: {td_loss:.5e} ",
            end="",
        )
        if epoch % (epo // 100) == 0:
            print()
        if epoch % 1000 == 0:
            with torch.no_grad():
                test(prnet, fin=True, epo=epoch)
    '''
    epo = 1000100
    for epoch in range(epo):
        total_loss = 0.
        tp_loss = 0.
        td_loss = 0.
        for batch_idx, (batch_input, batch_target) in enumerate(dataloader):
            batch_input, batch_target = batch_input[:, None, ...].to(torch.float32).cuda(), batch_target[:, None, ...].to(torch.float32).cuda()
            # print('batch_input: ', batch_input.shape)
            # print('batch_target: ', batch_target.shape)
            optimizer.zero_grad()
            pred = prnet.data_driven_forward(batch_input)
            # print('pred: ', pred.shape)

            phy_loss = prnet.get_phy_loss_data(pred)
            phy_loss = loss_fn(phy_loss, torch.zeros_like(phy_loss))
            data_loss = loss_fn(pred[:, -1:, ...], batch_target)
            # print('phy_loss: ', phy_loss.shape)
            # print('data_loss: ', data_loss.shape)
            # print(data_loss)

            loss = 1e3*phy_loss + 1e9*data_loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            tp_loss += phy_loss.item()
            td_loss += data_loss.item()

        print(
                f"\r seed: {seed} interval: {interval} It: {epoch} Loss: {total_loss:.5e} phy_loss: {tp_loss:.5e} data_loss: {td_loss:.5e} ",
            end="",
        )
        if epoch % (epo // 100) == 0:
            print()
        if epoch % 1000 == 0:
            with torch.no_grad():
                test(prnet, fin=True, epo=epoch)
    '''
