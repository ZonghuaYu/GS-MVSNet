import os, cv2, argparse, time
import numpy as np
from plyfile import PlyData, PlyElement

from data_io import read_pfm, save_pfm, read_pairfile, read_img, read_cam_file, save_mask


def filter(dataset_root, scan, img_folder, cam_folder,
           eval_folder, filter_folder, photo_threshold=0.8, dist_threshold=1, diff_threshold=0.01, geo_check_view=3):
    """
    # 1. filtered with photo consistency and geo consistency
    # 2. map to world location,and save to ply
    @param dataset_root: input folder:scene,include camera and photo
    @param scan:
    @param img_folder:
    @param cam_folder:
    @param eval_folder: include depth_est,photo consistency
    @param filter_folder: output folder:mask, depth_filter,
    @param photo_threshold:
    @param diff_threshold:
    """

    scan_location = os.path.join(dataset_root, scan)
    pair_path = os.path.join(scan_location, "pair.txt")

    eval_location = os.path.join(eval_folder, scan)

    filter_workspace = os.path.join(eval_location, filter_folder)
    os.makedirs(filter_workspace, exist_ok=True)

    vertexs, vertex_colors = [], []
    num_viewpoint, pairs = read_pairfile(pair_path)
    for ref_view, src_views in pairs:
        start_time = time.time()

        cam_path = os.path.join(scan_location, cam_folder,'{:0>8}_cam.txt'.format(ref_view))
        refimg_path = os.path.join(scan_location, img_folder,'{:0>8}.jpg'.format(ref_view))

        ref_intrinsics, ref_extrinsics = read_cam_file(cam_path)
        ref_img = read_img(refimg_path)

        depth_est_path = os.path.join(eval_location, "depth_est",'{:0>8}'.format(ref_view) + '.pfm')
        confidence_path = os.path.join(eval_location, "confidence",'{:0>8}'.format(ref_view) + '.pfm')
        ref_depth_est, scale = read_pfm(depth_est_path)
        confidence, _ = read_pfm(confidence_path)
        h, w = confidence.shape

        all_srcview_depth_ests, all_srcview_x, all_srcview_y, all_srcview_geomask = [], [], [], []
        # compute the geometric mask
        geo_mask_sum = 0
        for src_view in src_views:
            src_cam_path=os.path.join(scan_location, cam_folder, '{:0>8}_cam.txt'.format(src_view))
            src_depth_est=os.path.join(eval_location, "depth_est", '{:0>8}'.format(src_view) + '.pfm')

            src_intrinsics, src_extrinsics = read_cam_file(src_cam_path)
            src_depth_est = read_pfm(src_depth_est)[0]

            # check geometric consistency
            geo_mask, depth_reprojected, x2d_src, y2d_src = \
                check_geometric_consistency(ref_depth_est, ref_intrinsics,ref_extrinsics,
                                            src_depth_est,src_intrinsics, src_extrinsics, dist_threshold, diff_threshold)

            geo_mask_sum += geo_mask.astype(np.int32)
            all_srcview_depth_ests.append(depth_reprojected)
            all_srcview_x.append(x2d_src)
            all_srcview_y.append(y2d_src)
            all_srcview_geomask.append(geo_mask)


        depth_est_averaged = (sum(all_srcview_depth_ests) + ref_depth_est) / (geo_mask_sum + 1)

        # at least 3 source views matched
        geo_mask = geo_mask_sum >= geo_check_view
        photo_mask = confidence > photo_threshold   # photo consistency
        final_mask = np.logical_and(photo_mask, geo_mask)

        print("processing {}, ref-view{:0>2}, photo/geo/final-mask:{}/{}/{}".
              format(scan_location, ref_view, photo_mask.sum(), geo_mask.sum(), final_mask.sum()),
              " time:",time.time()-start_time)

        # save mask
        save_mask(os.path.join(filter_workspace, "{:0>8}_photo.png".format(ref_view)), photo_mask)
        save_mask(os.path.join(filter_workspace, "{:0>8}_geo.png".format(ref_view)), geo_mask)
        save_mask(os.path.join(filter_workspace, "{:0>8}_final.png".format(ref_view)), final_mask)

        save_pfm(os.path.join(filter_workspace, "{}".format(ref_view)+"_" + "depth_est.pfm"),
                                ref_depth_est * final_mask.astype(np.float32))

        ########################################################################################
        # 2.map to world location,and save to ply
        height, width = depth_est_averaged.shape[:2]
        valid_points = final_mask

        x, y = np.meshgrid(np.arange(0, width), np.arange(0, height))
        x, y, depth = x[valid_points], y[valid_points], depth_est_averaged[valid_points]

        color = ref_img[:h, :w, :][valid_points]

        # pix to camera ,to world
        xyz_ref = np.matmul(np.linalg.inv(ref_intrinsics),
                            np.vstack((x, y, np.ones_like(x))) * depth)
        xyz_world = np.matmul(np.linalg.inv(ref_extrinsics),
                              np.vstack((xyz_ref, np.ones_like(x))))[:3]
        vertexs.append(xyz_world.transpose((1, 0)))
        vertex_colors.append((color * 255).astype(np.uint8))

    vertexs = np.concatenate(vertexs, axis=0)
    vertex_colors = np.concatenate(vertex_colors, axis=0)
    vertexs = np.array([tuple(v) for v in vertexs], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    vertex_colors = np.array([tuple(v) for v in vertex_colors], dtype=[('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])

    vertex_all = np.empty(len(vertexs), vertexs.dtype.descr + vertex_colors.dtype.descr)
    for prop in vertexs.dtype.names:
        vertex_all[prop] = vertexs[prop]
    for prop in vertex_colors.dtype.names:
        vertex_all[prop] = vertex_colors[prop]

    el = PlyElement.describe(vertex_all, 'vertex')
    PlyData([el]).write(os.path.join(eval_location, scan+"_filters.ply"))
    print("saving the final model to", os.path.join(eval_location, scan+"_filters.ply"))

def check_geometric_consistency(depth_ref, intrinsics_ref, extrinsics_ref,
                                depth_src, intrinsics_src, extrinsics_src, dist_threshold=1, diff_threshold=0.01):
    width, height = depth_ref.shape[1], depth_ref.shape[0]
    x_ref, y_ref = np.meshgrid(np.arange(0, width), np.arange(0, height))
    depth_reprojected, x2d_reprojected, y2d_reprojected, x2d_src, y2d_src = reproject_with_depth(depth_ref, intrinsics_ref, extrinsics_ref,
                                                     depth_src, intrinsics_src, extrinsics_src)
    # check |p_reproj-p_1| < 1
    dist = np.sqrt((x2d_reprojected - x_ref) ** 2 + (y2d_reprojected - y_ref) ** 2)

    # check |d_reproj-d_1| / d_1 < 0.01
    depth_diff = np.abs(depth_reprojected - depth_ref)
    relative_depth_diff = depth_diff / depth_ref

    mask = np.logical_and(dist < dist_threshold, relative_depth_diff < diff_threshold)
    depth_reprojected[~mask] = 0

    return mask, depth_reprojected, x2d_src, y2d_src

# project the reference point cloud into the source view, then project back
def reproject_with_depth(depth_ref, intrinsics_ref, extrinsics_ref, depth_src, intrinsics_src, extrinsics_src):
    width, height = depth_ref.shape[1], depth_ref.shape[0]
    ## step1. project reference pixels to the source view
    # reference view x, y
    x_ref, y_ref = np.meshgrid(np.arange(0, width), np.arange(0, height))
    x_ref, y_ref = x_ref.reshape([-1]), y_ref.reshape([-1])
    # reference 3D space
    xyz_ref = np.matmul(np.linalg.inv(intrinsics_ref),
                        np.vstack((x_ref, y_ref, np.ones_like(x_ref))) * depth_ref.reshape([-1]))
    # source 3D space
    xyz_src = np.matmul(np.matmul(extrinsics_src, np.linalg.inv(extrinsics_ref)),
                        np.vstack((xyz_ref, np.ones_like(x_ref))))[:3]
    # source view x, y
    K_xyz_src = np.matmul(intrinsics_src, xyz_src)
    xy_src = K_xyz_src[:2] / K_xyz_src[2:3]

    ## step2. reproject the source view points with source view depth estimation
    # find the depth estimation of the source view
    x_src = xy_src[0].reshape([height, width]).astype(np.float32)
    y_src = xy_src[1].reshape([height, width]).astype(np.float32)
    sampled_depth_src = cv2.remap(depth_src, x_src, y_src, interpolation=cv2.INTER_LINEAR)

    # source 3D space
    # NOTE that we should use sampled source-view depth_here to project back
    xyz_src = np.matmul(np.linalg.inv(intrinsics_src),
                        np.vstack((xy_src, np.ones_like(x_ref))) * sampled_depth_src.reshape([-1]))
    # reference 3D space
    xyz_reprojected = np.matmul(np.matmul(extrinsics_ref, np.linalg.inv(extrinsics_src)),
                                np.vstack((xyz_src, np.ones_like(x_ref))))[:3]
    # source view x, y, depth
    depth_reprojected = xyz_reprojected[2].reshape([height, width]).astype(np.float32)
    K_xyz_reprojected = np.matmul(intrinsics_ref, xyz_reprojected)
    xy_reprojected = K_xyz_reprojected[:2] / K_xyz_reprojected[2:3]
    x_reprojected = xy_reprojected[0].reshape([height, width]).astype(np.float32)
    y_reprojected = xy_reprojected[1].reshape([height, width]).astype(np.float32)

    return depth_reprojected, x_reprojected, y_reprojected, x_src, y_src



if __name__=="__main__":

    parser = argparse.ArgumentParser(description='filter to maks cloudpoints...')
    parser.add_argument("-e",'--eval_folder', default="/hy-tmp/outputs", type=str, help='eval output location')#"../../outputs"
    parser.add_argument("-r",'--root_folder', default="/root", type=str, help='dataset root location')  #os.path.join("/home", "user22", "yu", "mvs_datasets")
    parser.add_argument('-f','--filter_folder', default="filter", type=str, help='filter output location')
    parser.add_argument('-t', '--dataset', default='dtu', type=str, help='dtu or tanks')

    args = parser.parse_args()
    print(args)

    root_dir = args.root_folder

    if args.dataset == "dtu":
        dataset_root = os.path.join(root_dir, "dtu")
        dtu_labels = [1, 4, 9, 10, 11, 12, 13, 15, 23, 24, 29, 32, 33, 34, 48, 49, 62, 75, 77, 110, 114, 118]
        scans = ["scan"+str(label) for label in dtu_labels]
        #nviewss = [49,]*len(dtu_labels)
        img_folder = "images"
        cam_folder = "cams"
        filter_args = {}
        for scan in scans:
            filter_args[scan] = {"photo_thresh": 0.8, "dist_threshold": 1, "diff_threshold": 0.01, "geo_check_views": 3}

    elif args.dataset == "tanks":
        dataset_root = os.path.join(root_dir, "TankandTemples", "intermediate")
        scans = ['Family','Francis', 'Horse', 'Lighthouse', 'M60', 'Panther', 'Playground', 'Train']
        #nviewss = [152, 302, 151, 309, 313, 314, 307, 301]
        img_folder =  "images"
        cam_folder = "cams_1"
        filter_args = {
            'Family':{"photo_thresh": 0.8, "dist_threshold": 1, "diff_threshold": 0.01, "geo_check_views": 5},
            'Francis': {"photo_thresh": 0.8, "dist_threshold": 0.5, "diff_threshold": 0.005, "geo_check_views": 6},
            'Horse': {"photo_thresh": 0.8, "dist_threshold": 1, "diff_threshold": 0.01, "geo_check_views": 5},
            'Lighthouse': {"photo_thresh": 0.8, "dist_threshold": 0.5, "diff_threshold": 0.005, "geo_check_views": 6},
            'M60': {"photo_thresh": 0.8, "dist_threshold": 1, "diff_threshold": 0.01, "geo_check_views": 5},
            'Panther': {"photo_thresh": 0.8, "dist_threshold": 1, "diff_threshold": 0.005, "geo_check_views": 6},
            'Playground': {"photo_thresh": 0.8, "dist_threshold": 1, "diff_threshold": 0.01, "geo_check_views": 6},
            'Train': {"photo_thresh": 0.8, "dist_threshold": 1, "diff_threshold": 0.01, "geo_check_views": 5},
        }

    else:
        print("please use dtu or tanks dataset,exit!")
        exit()

    # filter with photometric confidence maps and geometric constraints,then convert to world position
    for scan in scans:
        filter_arg = filter_args[scan]
        photo_thresh, dist_threshold, diff_threshold, geo_check_views = \
            filter_arg["photo_thresh"], filter_arg["dist_threshold"], filter_arg["diff_threshold"], filter_arg["geo_check_views"]

        print("scan:", scan, " photo_thresh:", photo_thresh, " dist_threshold:", dist_threshold,
              " diff_threshold:", diff_threshold, " geo_check_view:", geo_check_views)

        start_time = time.time()
        filter(dataset_root, scan, img_folder, cam_folder,
           args.eval_folder, args.filter_folder, photo_thresh, dist_threshold, diff_threshold, geo_check_views)

        print("all time:", time.time()-start_time)




