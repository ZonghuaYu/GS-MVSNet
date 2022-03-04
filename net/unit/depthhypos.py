import torch
import torch.nn as nn
import torch.nn.functional as F


class Depthhypos(nn.Module):
    def __init__(self, ):
        super(Depthhypos, self).__init__()

        # self.depth_limit_ratio = torch.nn.Parameter(torch.tensor(1/8), requires_grad=True)

        self.ratio_p = torch.nn.Parameter(torch.tensor(0.2), requires_grad=True)
        self.ratio_n = torch.nn.Parameter(torch.tensor(0.2), requires_grad=True)
        self.ratio_u = torch.nn.Parameter(torch.tensor(0.5), requires_grad=True)
        self.ratio_l = torch.nn.Parameter(torch.tensor(0.5), requires_grad=True)

        self.ratio_dlimit = torch.nn.Parameter(torch.tensor(1/8), requires_grad=True)
        self.ratio_ulimit = torch.nn.Parameter(torch.tensor(1/4), requires_grad=True)

        self.compensate_ratio = torch.nn.Parameter(torch.tensor(0.05), requires_grad=True)


    def forward(self, depth, ndepths, depth_range, iteration, level=None, intrinsics=None, extrinsics=None):
        """
        Depth hypothesis using four direction gradient
        @param depth: before level depth map(B,H,W) , in iteration=0 is None
        @param ndepths:
        @param iteration: depthhypos_limit = depth_limit_ratio * (depth_max - depth_min) / pow(2, iteration - 1)
        @param depth_range:
        @return: (B,ndepths,H,W)
        """
        nbatchs = depth_range.shape[0]
        depth_min, depth_max = depth_range[0].float()   # dtu tarin use the same depth range, and the batch size of eval is 1 ,so [0]

        if depth is None:
            depth_interval = (depth_max - depth_min) / (ndepths - 1)

            return torch.arange(depth_min, depth_max+1e-4, depth_interval) \
                        .view(1, ndepths, 1, 1).repeat(nbatchs, 1, 1, 1)

        # active paras
        compensate_ratio = self._active(self.compensate_ratio, 0, 0.2)
        ratio_dlimit, ratio_ulimit = self._active(self.ratio_dlimit, 0, 0.2), self._active(self.ratio_ulimit, 0, 0.5)
        ratio_p, ratio_n, ratio_u, ratio_l = \
            self._active(self.ratio_p), self._active(self.ratio_n), self._active(self.ratio_u), self._active(self.ratio_l)

        depth_limit = (depth_max - depth_min) / pow(2, iteration - 1)
        depthhypos_dlimit, depthhypos_ulimit  = ratio_dlimit * depth_limit, ratio_ulimit * depth_limit
        # print(compensate_ratio, ratio_ulimit, ratio_p, ratio_n, ratio_u, ratio_l, depthhypos_dlimit, depthhypos_ulimit)

        #cal min, max grad
        depth = torch.nn.functional.interpolate(depth.unsqueeze(1), scale_factor=2, mode='bicubic', align_corners=None)     #B, _, H, W
        with torch.no_grad():
            res_min, res_max = self._calGradminmax(depth)

        # keep the hypos range in a appropriate region
        mask_upper = (res_max - res_min) > ratio_ulimit * depthhypos_ulimit
        res_min[mask_upper], res_max[mask_upper] =\
            -1 * ratio_u * depthhypos_ulimit, (1-ratio_u) * depthhypos_ulimit

        # min >= 0 , max <=0, lower
        mask_npl = (res_min >= 0) | (res_max <= 0) | ((res_max - res_min) < depthhypos_dlimit)
        res_min[mask_npl], res_max[mask_npl] = \
            -1 * ratio_u * depthhypos_dlimit, (1 - ratio_u) * depthhypos_dlimit

        res_min, res_max = res_min * (1 + compensate_ratio), res_max * (1 + compensate_ratio)

        hypo_ranges = res_max - res_min
        intervals = hypo_ranges / (ndepths - 1)

        # generate depth hypos
        depth_hypos = (depth + res_min.unsqueeze(1)).repeat(1, ndepths, 1, 1)  # res_min<0
        for d in range(ndepths):
            depth_hypos[:, d, :, :] += intervals * d

        return depth_hypos


    def _calGradminmax(self, depth, ngrads=4):
        # left, right, up, down, padding=(nleft, nright, nup, ndown)
        res_l = torch.nn.ZeroPad2d(padding=(1, 0, 0, 0))(depth[:, :, :, 1:] - depth[:, :, :, :-1])
        res_r = torch.nn.ZeroPad2d(padding=(0, 1, 0, 0))(depth[:, :, :, :-1] - depth[:, :, :, 1:])
        res_u = torch.nn.ZeroPad2d(padding=(0, 0, 1, 0))(depth[:, :, 1:, :] - depth[:, :, :-1, :])
        res_d = torch.nn.ZeroPad2d(padding=(0, 0, 0, 1))(depth[:, :, :-1, :] - depth[:, :, 1:, :])

        # >> borrow the dim 1, it is channels=1, seek the min,max depth
        # >> * -1 convert the value, around - center
        res = torch.cat((res_l, res_r, res_u, res_d,), dim=1) * -1

        if ngrads == 8:
            res_lu = torch.nn.ZeroPad2d(padding=(1, 0, 1, 0))(depth[:, :, 1:, 1:] - depth[:, :, :-1, :-1])
            res_ld = torch.nn.ZeroPad2d(padding=(1, 0, 0, 1))(depth[:, :, -1:, 1:] - depth[:, :, 1:, :-1])
            res_ru = torch.nn.ZeroPad2d(padding=(0, 1, 1, 0))(depth[:, :, 1:, :-1] - depth[:, :, :-1, 1:])
            res_rd = torch.nn.ZeroPad2d(padding=(0, 1, 0, 1))(depth[:, :, :-1, :-1] - depth[:, :, 1:, 1:])

            # res = torch.cat((res_lu, res_ld, res_ru, res_rd, ), dim=1) * -1
            res = torch.cat((res_l, res_r, res_u, res_d, res_lu, res_ld, res_ru, res_rd), dim=1) * -1

        res_min, res_max = torch.min(res, dim=1)[0], torch.max(res, dim=1)[0]

        return res_min, res_max

    def _active(self, x, a=0, b=1):
        if x<=a:
            return torch.tensor(a+0.001).to(x.device)
        elif x<b:
            return x
        else:
            return torch.tensor(b-0.001).to(x.device)

"""
0
"""
# """
# grad search
# """
# class Depthhypos(nn.Module):
#     def __init__(self, ):
#         super(Depthhypos, self).__init__()
#
#         self.depth_limit_ratio = torch.nn.Parameter(torch.tensor(1/8), requires_grad=True)
#
#         self.ratio_p = torch.nn.Parameter(torch.tensor(0.2), requires_grad=True)
#         self.ratio_n = torch.nn.Parameter(torch.tensor(0.2), requires_grad=True)
#         self.ratio_u = torch.nn.Parameter(torch.tensor(0.5), requires_grad=True)
#         self.ratio_l = torch.nn.Parameter(torch.tensor(0.5), requires_grad=True)
#         self.ratio_ulimit = torch.nn.Parameter(torch.tensor(2.0), requires_grad=True)
#         self.compensate_ratio = torch.nn.Parameter(torch.tensor(0.05), requires_grad=True)
#
#
#     def forward(self, depth, ndepths, depth_range, iteration, level=None, intrinsics=None, extrinsics=None):
#         """
#         Depth hypothesis using four direction gradient
#         @param depth: before level depth map(B,H,W) , in iteration=0 is None
#         @param ndepths:
#         @param iteration: depthhypos_limit = depth_limit_ratio * (depth_max - depth_min) / pow(2, iteration - 1)
#         @param depth_range:
#         @return: (B,ndepths,H,W)
#         """
#         nbatchs = depth_range.shape[0]
#         depth_min, depth_max = depth_range[0].float()   # dtu tarin use the same depth range, and the batch size of eval is 1 ,so [0]
#
#         if depth is None:
#             depth_interval = (depth_max - depth_min) / (ndepths - 1)
#
#             return torch.arange(depth_min, depth_max+1e-4, depth_interval) \
#                         .view(1, ndepths, 1, 1).repeat(nbatchs, 1, 1, 1)
#
#         # active paras
#         depth_limit_ratio, compensate_ratio = \
#             self._active(self.depth_limit_ratio), self._active(self.compensate_ratio, 0, 0.2)
#         ratio_ulimit = self._active(self.ratio_ulimit, 1, 3)
#         ratio_p, ratio_n, ratio_u, ratio_l = \
#             self._active(self.ratio_p), self._active(self.ratio_n), self._active(self.ratio_u), self._active(self.ratio_l)
#
#         depthhypos_limit = depth_limit_ratio * (depth_max - depth_min) / pow(2, iteration - 1)
#         # print(depth_limit_ratio, compensate_ratio, ratio_ulimit, ratio_p, ratio_n, ratio_u, ratio_l)
#
#         #cal min, max grad
#         depth = torch.nn.functional.interpolate(depth.unsqueeze(1), scale_factor=2, mode='bicubic', align_corners=None)     #B, _, H, W
#         with torch.no_grad():
#             res_min, res_max = self._calGradminmax(depth)
#
#         # keep the hypos range in a appropriate region
#         mask_lower = (res_max - res_min) < depthhypos_limit
#         res_min[mask_lower], res_max[mask_lower] =\
#             -1 * ratio_l * depthhypos_limit, (1-ratio_l) * depthhypos_limit
#
#         mask_upper = (res_max - res_min) > ratio_ulimit * depthhypos_limit
#         res_min[mask_upper], res_max[mask_upper] =\
#             -1 * ratio_u * depthhypos_limit*ratio_ulimit, (1-ratio_u) * depthhypos_limit*ratio_ulimit
#
#         # min >= 0
#         mask_positive = res_min >= 0
#         res_min[mask_positive], res_max[mask_positive] =\
#             -1 * ratio_p * depthhypos_limit, (1-ratio_p) * depthhypos_limit
#
#         #max <=0
#         mask_negative = res_max <= 0
#         res_min[mask_negative], res_max[mask_negative] = \
#             -1 * (1 - ratio_n) * depthhypos_limit, ratio_n * depthhypos_limit
#
#         res_min, res_max = res_min * (1 + compensate_ratio), res_max * (1 + compensate_ratio)
#
#         hypo_ranges = res_max - res_min
#         intervals = hypo_ranges / (ndepths - 1)
#
#         # generate depth hypos
#         depth_hypos = (depth + res_min.unsqueeze(1)).repeat(1, ndepths, 1, 1)  # res_min<0
#         for d in range(ndepths):
#             depth_hypos[:, d, :, :] += intervals * d
#
#         return depth_hypos
#
#
#     def _calGradminmax(self, depth, ngrads=4):
#         # left, right, up, down, padding=(nleft, nright, nup, ndown)
#         res_l = torch.nn.ZeroPad2d(padding=(1, 0, 0, 0))(depth[:, :, :, 1:] - depth[:, :, :, :-1])
#         res_r = torch.nn.ZeroPad2d(padding=(0, 1, 0, 0))(depth[:, :, :, :-1] - depth[:, :, :, 1:])
#         res_u = torch.nn.ZeroPad2d(padding=(0, 0, 1, 0))(depth[:, :, 1:, :] - depth[:, :, :-1, :])
#         res_d = torch.nn.ZeroPad2d(padding=(0, 0, 0, 1))(depth[:, :, :-1, :] - depth[:, :, 1:, :])
#
#         # >> borrow the dim 1, it is channels=1, seek the min,max depth
#         # >> * -1 convert the value, around - center
#         res = torch.cat((res_l, res_r, res_u, res_d,), dim=1) * -1
#
#         if ngrads == 8:
#             res_lu = torch.nn.ZeroPad2d(padding=(1, 0, 1, 0))(depth[:, :, 1:, 1:] - depth[:, :, :-1, :-1])
#             res_ld = torch.nn.ZeroPad2d(padding=(1, 0, 0, 1))(depth[:, :, -1:, 1:] - depth[:, :, 1:, :-1])
#             res_ru = torch.nn.ZeroPad2d(padding=(0, 1, 1, 0))(depth[:, :, 1:, :-1] - depth[:, :, :-1, 1:])
#             res_rd = torch.nn.ZeroPad2d(padding=(0, 1, 0, 1))(depth[:, :, :-1, :-1] - depth[:, :, 1:, 1:])
#
#             # res = torch.cat((res_lu, res_ld, res_ru, res_rd, ), dim=1) * -1
#             res = torch.cat((res_l, res_r, res_u, res_d, res_lu, res_ld, res_ru, res_rd), dim=1) * -1
#
#         res_min, res_max = torch.min(res, dim=1)[0], torch.max(res, dim=1)[0]
#
#         return res_min, res_max
#
#     def _active(self, x, a=0, b=1):
#         if x<=a:
#             return torch.tensor(a+0.001).to(x.device)
#         elif x<b:
#             return x
#         else:
#             return torch.tensor(b-0.001).to(x.device)



"""
sight search
"""
def depthhypos_sight(depth, ndepths, depth_range, iteration=None, level=None, intrinsics=None, extrinsics=None, ):
    nbatchs = depth_range.shape[0]

    if depth is None:
        depth_min, depth_max = depth_range[0].float()
        depth_interval = (depth_max - depth_min) / (ndepths - 1)

        return torch.arange(depth_min, depth_max + 1e-4, depth_interval) \
            .view(1, ndepths, 1, 1).repeat(nbatchs, 1, 1, 1)

    depth = torch.nn.functional.interpolate(
        depth[None, :], size=None, scale_factor=2,mode='bicubic', align_corners=None).squeeze(0)  # False, recompute_scale_factor=False

    if nbatchs !=1:   # in train
        depth_hypos = cal_depthhypos_by_fixvalue(depth)
    else:
        intrinsics_tmp = intrinsics.clone()
        intrinsics_tmp[:, :, :2, :] = intrinsics_tmp[:, :, :2, :] / pow(2, level)
        depth_hypos = calDepthHypo(depth , intrinsics_tmp[:, 0, :, :], intrinsics_tmp[:, 1:, :, :],
                                   extrinsics[:, 0, :, :], extrinsics[:, 1:, :, :])
    return depth_hypos

def cal_depthhypos_by_fixvalue(depth, d=4):

    B, H, W = depth.shape
    depth_interval = torch.tensor([6.8085] * B).cuda()
    depth_hypos = depth.unsqueeze(1).repeat(1, d * 2, 1, 1)

    for depth_level in range(-d, d):
        depth_hypos[:, depth_level + d, :, :] += (depth_level) * depth_interval[0]

    return depth_hypos

def calDepthHypo(ref_depths , ref_intrinsics, src_intrinsics, ref_extrinsics, src_extrinsics,):
    """
    use: intrinsics_tmp[:, 0, :, :], intrinsics_tmp[:, 1:, :, :], extrinsics[:, 0, :, :], extrinsics[:, 1:, :, :]
    @param ref_depths:
    @param ref_intrinsics:
    @param src_intrinsics:
    @param ref_extrinsics:
    @param src_extrinsics:
    @return:
    """
    d = 4
    pixel_interval = 1

    nBatch = ref_depths.shape[0]
    height = ref_depths.shape[1]
    width = ref_depths.shape[2]

    with torch.no_grad():

        ref_depths = ref_depths
        ref_intrinsics = ref_intrinsics.double()
        src_intrinsics = src_intrinsics.squeeze(1).double()
        ref_extrinsics = ref_extrinsics.double()
        src_extrinsics = src_extrinsics.squeeze(1).double()

        interval_maps = []
        depth_hypos = ref_depths.unsqueeze(1).repeat(1, d * 2, 1, 1)
        for batch in range(nBatch):
            xx, yy = torch.meshgrid([torch.arange(0, width).cuda(), torch.arange(0, height).cuda()])

            xxx = xx.reshape([-1]).double()
            yyy = yy.reshape([-1]).double()

            X = torch.stack([xxx, yyy, torch.ones_like(xxx)], dim=0)

            D1 = torch.transpose(ref_depths[batch, :, :], 0, 1).reshape([-1])
            D2 = D1 + 1

            X1 = X * D1
            X2 = X * D2
            ray1 = torch.matmul(torch.inverse(ref_intrinsics[batch]), X1)
            ray2 = torch.matmul(torch.inverse(ref_intrinsics[batch]), X2)

            X1 = torch.cat([ray1, torch.ones_like(xxx).unsqueeze(0).double()], dim=0)
            X1 = torch.matmul(torch.inverse(ref_extrinsics[batch]), X1)
            X2 = torch.cat([ray2, torch.ones_like(xxx).unsqueeze(0).double()], dim=0)
            X2 = torch.matmul(torch.inverse(ref_extrinsics[batch]), X2)

            X1 = torch.matmul(src_extrinsics[batch][0], X1)
            X2 = torch.matmul(src_extrinsics[batch][0], X2)

            X1 = X1[:3]
            X1 = torch.matmul(src_intrinsics[batch][0], X1)
            X1_d = X1[2].clone()
            X1 /= X1_d

            X2 = X2[:3]
            X2 = torch.matmul(src_intrinsics[batch][0], X2)
            X2_d = X2[2].clone()
            X2 /= X2_d

            k = (X2[1] - X1[1]) / (X2[0] - X1[0])
            b = X1[1] - k * X1[0]

            theta = torch.atan(k)
            X3 = X1 + torch.stack(
                [torch.cos(theta) * pixel_interval, torch.sin(theta) * pixel_interval, torch.zeros_like(X1[2, :])], dim=0)

            A = torch.matmul(ref_intrinsics[batch], ref_extrinsics[batch][:3, :3])
            tmp = torch.matmul(src_intrinsics[batch][0], src_extrinsics[batch][0, :3, :3])
            A = torch.matmul(A, torch.inverse(tmp))

            tmp1 = X1_d * torch.matmul(A, X1)
            tmp2 = torch.matmul(A, X3)

            M1 = torch.cat([X.t().unsqueeze(2), tmp2.t().unsqueeze(2)], axis=2)[:, 1:, :]
            M2 = tmp1.t()[:, 1:]
            ans = torch.matmul(torch.inverse(M1), M2.unsqueeze(2))
            delta_d = ans[:, 0, 0]

            interval_maps = torch.abs(delta_d).mean().repeat(ref_depths.shape[2], ref_depths.shape[1]).t()

            for depth_level in range(-d, d):
                depth_hypos[batch, depth_level + d, :, :] += depth_level * interval_maps

        return depth_hypos.float()


if __name__=="__main__":
    pass
    x = torch.randn(2,5,5)
    print(x)

    y1 = torch.nn.functional.interpolate(x[None, :], size=None, scale_factor=2,
                                            mode='bicubic',align_corners=None).squeeze(0)
    y2 = torch.nn.functional.interpolate(x.unsqueeze(1), scale_factor=2,
                                         mode='bicubic', align_corners=None).squeeze(1)

    print(y2-y1)
