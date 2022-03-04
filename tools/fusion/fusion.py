import logging
import os,shutil,argparse,cv2
import conf
from tool import *


def probability_filter(depth_folder, prob_folder, nviews, prob_threshold):

    for view in range(nviews):
        init_depth_map_path = os.path.join(depth_folder,"{:08d}.pfm".format(view))
        prob_map_path = os.path.join(prob_folder, "{:08d}.pfm".format(view))
        out_depth_map_path = os.path.join(depth_folder, "{:08d}_prob_filtered.pfm".format(view))

        depth_map, _ = read_pfm(init_depth_map_path)
        prob_map, _ = read_pfm(prob_map_path)
        depth_map[prob_map < prob_threshold] = 0
        save_pfm(out_depth_map_path, depth_map)


def imgcam_convert(image_folder, cam_folder, fusibile_workspace, nviews):
    # output dir
    fusion_cam_folger = os.path.join(fusibile_workspace, 'cams')
    fusion_image_folder = os.path.join(fusibile_workspace, 'images')
    os.makedirs(fusion_cam_folger,exist_ok=True)
    os.makedirs(fusion_image_folder,exist_ok=True)

    # cal proj cameras: [KR KT]
    for view in range(nviews):
        in_cam_file = os.path.join(cam_folder, "{:08d}_cam.txt".format(view))
        out_cam_file = os.path.join(fusion_cam_folger, "{:08d}.png.P".format(view))

        cal_projection_matrix(in_cam_file, out_cam_file)

    # copy images to gipuma image folder
    for view in range(nviews):
        in_image_file = os.path.join(image_folder, "{:08d}.jpg".format(view))
        out_image_file = os.path.join(fusion_image_folder, "{:08d}.png".format(view))

        shutil.copy(in_image_file, out_image_file)

def to_gipuma(depth_folder, fusibile_workspace, nviews):
    gipuma_prefix = '2333__'
    for view in range(nviews):

        sub_depth_folder = os.path.join(fusibile_workspace, gipuma_prefix+"{:08d}".format(view))
        os.makedirs(sub_depth_folder, exist_ok=True)

        in_depth_pfm = os.path.join(depth_folder, "{:08d}_prob_filtered.pfm".format(view))
        out_depth_dmb = os.path.join(sub_depth_folder, 'disp.dmb')
        fake_normal_dmb = os.path.join(sub_depth_folder, 'normals.dmb')

        image, _ = read_pfm(in_depth_pfm)
        write_gipuma_dmb(out_depth_dmb, image)

        fake_gipuma_normal(out_depth_dmb, fake_normal_dmb)

def depth_map_fusion(fusibile_workspace, fusibile_exe_path, disp_thresh, num_consistent):

    cam_folder = os.path.join(fusibile_workspace, 'cams')
    image_folder = os.path.join(fusibile_workspace, 'images')
    depth_min = 0.001
    depth_max = 100000
    normal_thresh = 360

    cmd = fusibile_exe_path
    cmd = cmd + ' -input_folder ' + fusibile_workspace + '/'
    cmd = cmd + ' -p_folder ' + cam_folder + '/'
    cmd = cmd + ' -images_folder ' + image_folder + '/'
    cmd = cmd + ' --depth_min=' + str(depth_min)
    cmd = cmd + ' --depth_max=' + str(depth_max)
    cmd = cmd + ' --normal_thresh=' + str(normal_thresh)
    cmd = cmd + ' --disp_thresh=' + str(disp_thresh)
    cmd = cmd + ' --num_consistent=' + str(num_consistent)
    print (cmd)
    os.system(cmd)

    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='dtu fusion parameter setting')
    parser.add_argument("-c", '--cut', action='store_true', help='cut img to keep the same size with eval')
    parser.add_argument("-f", '--filter', action='store_true', help='filter depth_map with prob_map')
    parser.add_argument("-m", '--move', action='store_true', help='move img and cal proj matrix')
    parser.add_argument("-g", '--gipuma', action='store_true', help='convert depth maps and fake normal maps')
    parser.add_argument("-d", '--depth_fusion', action='store_true', help='depth map fusion with gipuma')

    parser.add_argument('-i', '--num_iters', default=5, type=int, help='number of iterations of the network')
    parser.add_argument('-t', '--dataset', default='dtu', type=str, help='dtu or tanks')

    args = parser.parse_args()
    print(args)

    if args.dataset == "dtu":
        eval_root = conf.dtu_eval_root
        scans = conf.dtu_scans
        nviewss = conf.dtu_nviews
        img_folder = conf.dtu_img_folder
        cam_folder = conf.dtu_cam_folder
        prob_threshold = conf.dtu_prob_threshold
        disp_threshold = conf.dtu_disp_threshold
        check_views = conf.dtu_check_views

    elif args.dataset == "tanks":
        eval_root = conf.tanks_eval_root
        scans = conf.tanks_scans
        nviewss = conf.tanks_nviews
        img_folder = conf.tanks_img_folder
        cam_folder = conf.tanks_cam_folder
        prob_threshold = conf.tanks_prob_threshold
        disp_threshold = conf.tanks_disp_threshold
        check_views = conf.tanks_check_views

    else:
        print("please use dtu or tanks dataset,exit!")
        exit()

    # A.Crop the picture to match the network input
    logging.info("A.Crop the picture to match the network input")
    if args.cut:
        for scan, nviews in zip(scans, nviewss):
            os.makedirs(os.path.join(eval_root, scan, conf.cut_img_folder), exist_ok=True)
            for vid in range(nviews):
                img_filename = os.path.join(eval_root, scan, img_folder, '{:0>8}.jpg'.format(vid))
                out_img_filename = os.path.join(eval_root, scan, conf.cut_img_folder, '{:0>8}.jpg'.format(vid))

                img = cv2.imread(img_filename)
                h, w, _ = img.shape
                h, w = cal_ncutpixs(h, w, args.num_iters)
                img = img[:h][:w]

                logging.info("save location:" + out_img_filename)
                cv2.imwrite(out_img_filename, img)

    # B.Fusion
    logging.info("B.Fusion")
    for scan, nviews, check_view in zip(scans, nviewss, check_views):
        logging.info("current scan:"+scan+" nviews:"+str(nviews)+" check_view:"+str(check_view))

        scan_folder = os.path.join(eval_root, scan)
        cam_folder = os.path.join(scan_folder, cam_folder)
        image_folder = os.path.join(scan_folder, conf.cut_img_folder)  # use crop img

        eval_dc_folder = os.path.join(conf.eval_folder, scan)
        depth_folder = os.path.join(eval_dc_folder, 'depth_est')
        prob_folder = os.path.join(eval_dc_folder, 'confidence')

        fusibile_workspace = os.path.join(conf.eval_folder, scan, "fuse")
        os.makedirs(fusibile_workspace, exist_ok=True)

        # probability filtering , save *_prob_filtered.pfm in depth_folder
        logging.info('>>1.filter depth map with probability map')
        if args.filter:
            probability_filter(depth_folder, prob_folder, nviews, prob_threshold)

        # data conversion
        logging.info('>>2.move img and cal proj matrix')
        if args.move:
            imgcam_convert(image_folder, cam_folder, fusibile_workspace, nviews)

        # convert depth maps and fake normal maps
        logging.info('>>3.convert depth maps and fake normal maps')
        if args.gipuma:
            to_gipuma(depth_folder, fusibile_workspace, nviews)

        # depth map fusion with gipuma
        logging.info('>>4.Run depth map fusion & filter')
        if args.depth_fusion:
            depth_map_fusion(fusibile_workspace, conf.fusibile_exe_path, disp_threshold, check_view)    # parser = argparse.ArgumentParser(description='Train parameter setting')
