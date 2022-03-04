import torch


def scale_imgcam(imgs, intrinsics, extrinsics, level):
    """
    zoom the img &cam matrixs in iter
    @param imgs: VIEW*(B,C,H,W)
    @param intrinsics: （B,VIEW,3,3）
    @param extrinsics:  (B,VIEW,4,4）
    @param level:  level = len(self.ndepths) - 1 - iteration  # 43210 10
    @return:
    """
    # # scale img
    for i in range(level):
        imgs = [torch.nn.functional.interpolate(
            img, scale_factor=0.5, mode='bilinear', align_corners=None).detach()
                for img in imgs]  # align_corners=False, recompute_scale_factor=False

    # scale intrinsics matrix & making proj matrix
    intrinsics_tmp, proj_matrix = intrinsics.clone(), extrinsics.clone()    # must clone!!!
    intrinsics_tmp[:, :, :2, :] = intrinsics_tmp[:, :, :2, :] / pow(2, level)
    proj_matrix[:, :, :3, :4] = torch.matmul(intrinsics_tmp, proj_matrix[:, :, :3, :4])
    proj_matrix = torch.unbind(proj_matrix, 1)  # VIEW*(B,4,4)
    ref_proj, src_projs = proj_matrix[0], proj_matrix[1:]

    return imgs, ref_proj, src_projs


if __name__=="__main__":
    pass