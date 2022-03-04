import torch


class GS_MVSNet(torch.nn.Module):
    def __init__(self, scalmodule, featuremodule, depthhyposmodule, homomodule,
                 regularmodule, regressmodule, ndepths, groups,):
        super(GS_MVSNet, self).__init__()

        self.scale = scalmodule
        self.feature = featuremodule
        self.depthhypos = depthhyposmodule
        self.homoaggregate = homomodule
        self.regular = regularmodule
        self.depth_regress, self.confidence_regress = regressmodule

        self.ndepths = ndepths
        self.groups = groups

    def set_niters(self, niters):
        self.ndepths = self.ndepths[:niters]

    def forward(self, imgs, extrinsics, intrinsics, depth_range):
        """
        Iterative search depth
        @param imgs: （B,VIEW,C,H,W） view0 is ref img
        @param extrinsics: （B,VIEW,4,4）
        @param intrinsics: （B,VIEW,3,3）
        @param depth_range: (B, 2) B*(depth_min, depth_max) dtu: [425.0, 1065.0] tanks: [-, -]
        @return:
        """
        origin_imgs = torch.unbind(imgs.float(), 1)  # VIEW*(B,C,H,W)

        depth, confidence = None, None
        for iteration, ndepths in enumerate(self.ndepths):
            level = len(self.ndepths) - 1 - iteration  # 4321 21 ->0 refine

            # 1. depth hypos
            depth_hypos = self.depthhypos(depth, ndepths, depth_range, iteration).to(intrinsics.device)

            # 2.scale img 、matrix & feature extraction
            imgs, ref_proj, src_projs = self.scale(origin_imgs, intrinsics, extrinsics, level)

            # 3. feature extraction
            features = [self.feature(img) for img in imgs]
            del imgs

            # 4. homo
            cost_volume = self.homoaggregate(features, ref_proj, src_projs, depth_hypos)
            del features, ref_proj, src_projs

            # 5. regular
            prob_volume = self.regular(cost_volume)  # (B,D,H,W)
            del cost_volume

            # 6. regress depth & confidence
            depth = self.depth_regress(prob_volume, depth_hypos)
            # confidence = self.confidence_regress(prob_volume, confidence)

        confidence = self.confidence_regress(prob_volume)
        return {"depth": depth, "confidence": confidence}


if __name__=="__main__":
    pass

