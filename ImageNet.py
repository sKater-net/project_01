import torch
from torch import nn
from torch.nn import functional as F
import math
from typing import *
from torch_geometric.nn import GraphConv

def edge_list(G):
    mask = G != -1.5
    list_e, cla = mask.nonzero(as_tuple=True)
    list_e = list_e.tolist()
    cla = cla.tolist()
    res = [list_e, cla]
    return res

def build_G_from_S(S, k=7, eplison=0.6):
    G = torch.ones(S.shape).cuda() * -1.5
    G_ = torch.where(S > eplison
                     , S.cuda(), torch.tensor(-1.50, dtype=torch.float).cuda()).cuda()
    for i in range(G_.shape[0]):
        idx = torch.argsort(-G_[i])[:k]
        G[i][idx] = G_[i][idx]
    del G_
    torch.cuda.empty_cache()
    return G

class ImageNet(nn.Module):
    def __init__(self, y_dim, bit, norm=True, mid_num1=1024, mid_num2=1024, hiden_layer=3):
        """
        :param y_dim: dimension of tags
        :param bit: bit number of the final binary code
        """
        super(ImageNet, self).__init__()
        self.module_name = "img_model"

        mid_num1 = mid_num1 if hiden_layer > 1 else bit
        modules = [FastKANLayer(y_dim, mid_num1)]
        if hiden_layer >= 2:
            modules += [nn.ReLU(inplace=True)]
            pre_num = mid_num1
            for i in range(hiden_layer - 2):
                if i == 0:
                    modules += [FastKANLayer(mid_num1, mid_num2), nn.ReLU(inplace=True)]
                else:
                    modules += [FastKANLayer(mid_num2, mid_num2), nn.ReLU(inplace=True)]
                pre_num = mid_num2
            modules += [FastKANLayer(pre_num, 1024)]
        self.fc = nn.Sequential(*modules)
        self.fc2 = GraphConv(1024, 1024)
        self.fc3 = GraphConv(1024, bit)
        self.norm = norm

    def forward(self, x):
        y = self.fc(x).tanh()
        x1 = F.normalize(x)
        S = x1.mm(x1.t())
        G = build_G_from_S(S,k=6, eplison=0.6)
        G = torch.LongTensor(edge_list(G)).cuda()
        out = self.fc2(y,G)
        if self.norm:
            norm_x = torch.norm(out, dim=1, keepdim=True)
            out = out / norm_x
        return out
    
class FastKANLayer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        grid_min: float = -3.,
        grid_max: float = 3.,
        num_grids: int = 4,
        use_base_update: bool = True,
        use_layernorm: bool = True,
        base_activation = F.silu,
        spline_weight_init_scale: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layernorm = None
        if use_layernorm:
            assert input_dim > 1, "Do not use layernorms on 1D inputs. Set `use_layernorm=False`."
            self.layernorm = nn.LayerNorm(input_dim)
        self.rbf = RadialBasisFunction(grid_min, grid_max, num_grids)
        self.spline_linear = SplineLinear(input_dim * num_grids, output_dim, spline_weight_init_scale)
        self.use_base_update = use_base_update
        if use_base_update:
            self.base_activation = base_activation
            self.base_linear = nn.Linear(input_dim, output_dim)

    def forward(self, x, use_layernorm=True):
        if self.layernorm is not None and use_layernorm:
            spline_basis = self.rbf(self.layernorm(x))
        else:
            spline_basis = self.rbf(x)
        ret = self.spline_linear(spline_basis.view(*spline_basis.shape[:-2], -1))
        if self.use_base_update:
            base = self.base_linear(self.base_activation(x))
            ret = ret + base
        return ret

class RadialBasisFunction(nn.Module):
    def __init__(
        self,
        grid_min: float = -4.,
        grid_max: float = 4.,
        num_grids: int = 4,
        denominator: float = None,  # larger denominators lead to smoother basis
    ):
        super().__init__()
        self.grid_min = grid_min
        self.grid_max = grid_max
        self.num_grids = num_grids
        grid = torch.linspace(grid_min, grid_max, num_grids)
        self.grid = torch.nn.Parameter(grid, requires_grad=False)
        self.denominator = denominator or (grid_max - grid_min) / (num_grids - 1)

    def forward(self, x):
        return torch.exp(-((x[..., None] - self.grid) / self.denominator) ** 2)

class SplineLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, init_scale: float = 0.1, **kw) -> None:
        self.init_scale = init_scale
        super().__init__(in_features, out_features, bias=False, **kw)

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.weight, mean=0, std=self.init_scale)

# class ImageNet(nn.Module):
#     def __init__(self, y_dim, bit, norm=True, mid_num1=1024*8, mid_num2=1024*8, hiden_layer=3):
#         """
#         :param y_dim: dimension of tags
#         :param bit: bit number of the final binary code
#         """
#         super(ImageNet, self).__init__()
#         self.module_name = "img_model"

#         mid_num1 = mid_num1 if hiden_layer > 1 else bit
#         modules = [nn.Linear(y_dim, mid_num1)]
#         if hiden_layer >= 2:
#             modules += [nn.ReLU(inplace=True)]
#             pre_num = mid_num1
#             for i in range(hiden_layer - 2):
#                 if i == 0:
#                     modules += [nn.Linear(mid_num1, mid_num2), nn.ReLU(inplace=True)]
#                 else:
#                     modules += [nn.Linear(mid_num2, mid_num2), nn.ReLU(inplace=True)]
#                 pre_num = mid_num2
#             modules += [nn.Linear(pre_num, bit)]
#         self.fc = nn.Sequential(*modules)
#         self.norm = norm

#     def forward(self, x):
#         out = self.fc(x).tanh()
#         if self.norm:
#             norm_x = torch.norm(out, dim=1, keepdim=True)
#             out = out / norm_x
#         return out