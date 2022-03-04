import torch
import numpy as np
from tools import data_io
from load.getpath import get_img_path,get_cam_path,get_depth_path


class LoadDataset(torch.utils.data.Dataset):
    def __init__(self,datasetpath,pairpath,scencelist,lighting_label,nviews):
        super(LoadDataset, self).__init__()
        self.datasetpath = datasetpath
        self.scenelist = scencelist
        # self.mode = mode
        self.lighting_label = lighting_label
        self.nviews = nviews

        self.num_viewpoint,self.pairs = data_io.read_pairfile(pairpath)
        self.all_compose = self._copmose_input()

    def __getitem__(self, item):
        scene,lighting,ref_view, src_views = self.all_compose[item]
        rs_views = [ref_view] + src_views[:self.nviews - 1]

        imgs, mask, depth, extrinsics, intrinsics = [], None, None, [], []
        scan_folder = "scan{}_train".format(scene)

        for i, vid in enumerate(rs_views):

            img_filename = get_img_path(self.datasetpath, scan_folder, vid, lighting, mode="train")
            cam_filename = get_cam_path(self.datasetpath, scan_folder, vid, mode="train")

            imgs.append(data_io.read_img(img_filename))
            intrinsic, extrinsic = data_io.read_cam_file(cam_filename) # (3,3) (4,4)

            extrinsics.append(extrinsic)
            intrinsics.append(intrinsic)

            if i == 0:  # reference view
                depth_filename = get_depth_path(self.datasetpath, scan_folder, vid, mode="train")
                ref_depth = np.array(data_io.read_pfm(depth_filename)[0], dtype=np.float32)

        imgs = np.stack(imgs).transpose([0, 3, 1, 2])
        extrinsics = np.stack(extrinsics)
        intrinsics = np.stack(intrinsics)

        return {"imgs": imgs,
                "intrinsics": intrinsics,
                "extrinsics": extrinsics,
                "ref_depth": ref_depth,
                "depth_range": np.array([425.0, 1065.0])
                }

    def __len__(self):
        # nscans* nviews（49）* nlightings（7）
        return len(self.scenelist)*len(self.pairs)*len(self.lighting_label)

    def _copmose_input(self):
        all_compose = []
        for scene in self.scenelist:
            for r,s in self.pairs:
                for lighting in self.lighting_label:
                    all_compose.append([scene,lighting,r,s])

        return all_compose


if __name__=="__main__":
    pass
