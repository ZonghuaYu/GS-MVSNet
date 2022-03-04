import torch
import torch.nn as nn
import torch.nn.functional as F

"""
variance method
"""
def homo_aggregate_by_variance(features, ref_proj, src_projs, depth_hypos):

    ndepths = depth_hypos.shape[1]
    ref_feature, src_features = features[0], features[1:]  # (B,C,H,W),(nviews-1)*（B,C,H,W）
    ref_volume = ref_feature.unsqueeze(2).repeat(1, 1, ndepths, 1, 1)  # （B,C,D,H,W）

    volume_sum = ref_volume
    volume_sq_sum = ref_volume**2

    for src_fea, src_proj in zip(src_features, src_projs):
        # torch.cuda.empty_cache()
        warped_volume = homo_warping(src_fea, src_proj, ref_proj, depth_hypos)
        volume_sum = volume_sum + warped_volume
        volume_sq_sum = volume_sq_sum + warped_volume ** 2
        del warped_volume

    cost_volume = volume_sq_sum.div_(len(src_features)+1).sub_(volume_sum.div_(len(src_features)+1).pow_(2))
    del volume_sum, volume_sq_sum

    return cost_volume

def homo_warping(src_fea, src_proj, ref_proj, depth_hypos):
    """

    @param src_fea: [B, C, H, W]
    @param src_proj: [B, 4, 4]
    @param ref_proj: [B, 4, 4]
    @param depth_hypos: [B, Ndepth, 1, 1] or [B,NDepths,H,W]
    @return: [B, C, Ndepth, H, W]
    """
    batch, ndepths, H, W = depth_hypos.shape
    batch, channels ,height, width= src_fea.shape   #torch.Size([1, 32, 128, 160])

    with torch.no_grad():
        proj = torch.matmul(src_proj, torch.inverse(ref_proj))
        rot = proj[:, :3, :3]  # [B,3,3]
        trans = proj[:, :3, 3:4]  # [B,3,1]

        y, x = torch.meshgrid([torch.arange(0, height, dtype=torch.float32, device=src_fea.device),
                               torch.arange(0, width, dtype=torch.float32, device=src_fea.device)])
        y, x = y.contiguous(), x.contiguous()
        y, x = y.view(height * width), x.view(height * width)
        xyz = torch.stack((x, y, torch.ones_like(x)))  # [3, H*W]
        xyz = torch.unsqueeze(xyz, 0).repeat(batch, 1, 1)  # [B, 3, H*W]
        # del x, y

        rot_xyz = torch.matmul(rot, xyz)  # [B, 3, H*W]

        rot_depth_xyz = rot_xyz.unsqueeze(2).repeat(1, 1, ndepths, 1) * depth_hypos.view(batch, 1, ndepths, H * W) # [B, 3, Ndepth, H*W]

        proj_xyz = rot_depth_xyz + trans.view(batch, 3, 1, 1)  # [B, 3, Ndepth, H*W]
        proj_xy = proj_xyz[:, :2, :, :] / proj_xyz[:, 2:3, :, :]  # [B, 2, Ndepth, H*W]

        proj_x_normalized = proj_xy[:, 0, :, :] / ((width - 1) / 2) - 1
        proj_y_normalized = proj_xy[:, 1, :, :] / ((height - 1) / 2) - 1
        proj_xy = torch.stack((proj_x_normalized, proj_y_normalized), dim=3)  # [B, Ndepth, H*W, 2]
        # del rot_xyz, rot_depth_xyz, proj_xyz, proj_x_normalized, proj_y_normalized

    warped_src_fea = \
        F.grid_sample(src_fea, proj_xy.view(batch, ndepths * height, width, 2), mode='bilinear',padding_mode='zeros')
    warped_src_fea = warped_src_fea.view(batch, channels, ndepths, height, width)

    return warped_src_fea


if __name__=="__main__":
    pass


