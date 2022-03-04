import torch
import torch.nn.functional as F


class Loss(torch.nn.Module):
    def __init__(self):
        super(Loss,self).__init__()

    def forward(self, pred, depth_gt, depth_range):

        mask = depth_gt > depth_range[0,0]    #depth_min
        loss_depth = F.smooth_l1_loss(pred[mask], depth_gt[mask], reduction='mean')

        return loss_depth

""""""""
# class Loss(torch.nn.Module):
#     def __init__(self, ):
#         super(Loss,self).__init__()
#
#     def forward(self, depth_est, depth_gt, depth_range):
#
#         mask = depth_gt > depth_range[0,0]    #depth_min
#         loss = F.smooth_l1_loss(depth_est[mask], depth_gt[mask], reduction='mean')
#
#         return loss

""""""""""""""""""""""""
#
# class Loss(torch.nn.Module):
#     def __init__(self, pix_loss=False):
#         super(Loss,self).__init__()
#         self.pix_loss = pix_loss
#
#     def forward(self, depth_est, depth_gt, depth_range, extrinsics=None, intrinsics=None, imgs=None):
#
#         mask = depth_gt > depth_range[0,0]    #depth_min
#         loss = F.smooth_l1_loss(depth_est[mask], depth_gt[mask], reduction='mean')
#
#         if self.pix_loss:
#             loss += 0.2*self._pix_loss(imgs, extrinsics, intrinsics, depth_est, mask)
#
#         return loss
#
#     def _pix_loss(self, imgs, extrinsics, intrinsics, depth_est, mask):
#
#         B, V, C, H, W = imgs.shape
#
#         origin_imgs = torch.unbind(imgs.float(), 1)  # VIEW*(B,C,H,W)
#         ref_img, src_imgs = origin_imgs[0], origin_imgs[1:]
#
#         proj_matrix = extrinsics.clone()
#         proj_matrix[:, :, :3, :4] = torch.matmul(intrinsics, proj_matrix[:, :, :3, :4])
#         proj_matrix = torch.unbind(proj_matrix, 1)  # VIEW*(B,4,4)
#         ref_proj, src_projs = proj_matrix[0], proj_matrix[1:]
#
#         mask = mask.view(B, 1, H, W).repeat(1, C ,1, 1)
#         feature_loss = 0.0
#         for src_img, src_proj in zip(src_imgs, src_projs):
#             warp_img = homo_warping(src_img, src_proj, ref_proj, depth_est)
#             mask_all = mask | (warp_img > 0)
#
#             feature_loss += F.smooth_l1_loss(ref_img[mask_all], warp_img[mask_all])
#
#         print("pix loss:",feature_loss/len(src_projs)*0.2)
#         return feature_loss/len(src_projs)
#
#
# def homo_warping(src_fea, src_proj, ref_proj, depth_hypos):
#     # """
#     #
#     # @param src_fea: [B, C, H, W]
#     # @param src_proj: [B, 4, 4]
#     # @param ref_proj: [B, 4, 4]
#     # @param depth_hypos: [B, Ndepth, 1, 1] or [B,NDepths,H,W]
#     # @return: [B, C, Ndepth, H, W]
#     # """
#     ndepths = 1
#     batch, H, W = depth_hypos.shape
#     batch, channels ,height, width= src_fea.shape   #torch.Size([1, 32, 128, 160])
#
#     with torch.no_grad():
#         proj = torch.matmul(src_proj, torch.inverse(ref_proj))
#         rot = proj[:, :3, :3]  # [B,3,3]
#         trans = proj[:, :3, 3:4]  # [B,3,1]
#
#         y, x = torch.meshgrid([torch.arange(0, height, dtype=torch.float32, device=src_fea.device),
#                                torch.arange(0, width, dtype=torch.float32, device=src_fea.device)])
#         y, x = y.contiguous(), x.contiguous()
#         y, x = y.view(height * width), x.view(height * width)
#         xyz = torch.stack((x, y, torch.ones_like(x)))  # [3, H*W]
#         xyz = torch.unsqueeze(xyz, 0).repeat(batch, 1, 1)  # [B, 3, H*W]
#         # del x, y
#
#         rot_xyz = torch.matmul(rot, xyz)  # [B, 3, H*W]
#
#         rot_depth_xyz = rot_xyz.unsqueeze(2).repeat(1, 1, ndepths, 1) * depth_hypos.view(batch, 1, ndepths, H * W) # [B, 3, Ndepth, H*W]
#
#         proj_xyz = rot_depth_xyz + trans.view(batch, 3, 1, 1)  # [B, 3, Ndepth, H*W]
#         proj_xy = proj_xyz[:, :2, :, :] / proj_xyz[:, 2:3, :, :]  # [B, 2, Ndepth, H*W]
#
#         proj_x_normalized = proj_xy[:, 0, :, :] / ((width - 1) / 2) - 1
#         proj_y_normalized = proj_xy[:, 1, :, :] / ((height - 1) / 2) - 1
#         proj_xy = torch.stack((proj_x_normalized, proj_y_normalized), dim=3)  # [B, Ndepth, H*W, 2]
#         # del rot_xyz, rot_depth_xyz, proj_xyz, proj_x_normalized, proj_y_normalized
#
#     warped_src_fea = F.grid_sample(src_fea, proj_xy.view(batch, ndepths * height, width, 2), mode='bilinear',
#                                    padding_mode='zeros')    #,align_corners=False
#     warped_src_fea = warped_src_fea.view(batch, channels, ndepths, height, width)
#
#     return warped_src_fea.squeeze(2)


if __name__=="__main__":
    pass

