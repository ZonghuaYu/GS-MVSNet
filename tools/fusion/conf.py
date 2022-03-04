import os


"""
 logging info format 
"""
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s-%(levelname)s: %(message)s")


"""
eval ouput depth and confidence location
"""
eval_folder = "../../outputs"   #eval output location


""" 
dataset args
"""
root_dir = os.path.join("/data", "user10")

#>>>>> dtu
dtu_eval_root = os.path.join(root_dir, "dtu")    #"/root/dtu"
dtu_img_folder = "images"
dtu_cam_folder = "cams"

dtu_labels = [1, 4, 9, 10, 11, 12, 13, 15, 23, 24, 29, 32, 33, 34, 48, 49, 62, 75, 77, 110, 114, 118]
dtu_scans = ["scan"+str(label) for label in dtu_labels]
dtu_nviews = [49,]*len(dtu_labels)


#>>>>> tanks
tanks_eval_root = os.path.join(root_dir, "TankandTemples", "intermediate")
tanks_img_folder = "images"
tanks_cam_folder = "cams_1"

tanks_scans = ['Family', 'Francis', 'Horse', 'Lighthouse', 'M60', 'Panther', 'Playground', 'Train']
tanks_nviews = [152, 302, 151, 309, 313, 314, 307, 301]

"""
cut img dir
"""
cut_img_folder = "images_cut"


"""
fusion parameters
"""
fusibile_exe_path = "/data/user10/fusibile/fusibile"
dtu_prob_threshold = 0.8
dtu_disp_threshold = 0.16
dtu_check_views = [3, ] * len(dtu_labels)

tanks_prob_threshold = 0.8
tanks_disp_threshold = 0.16
tanks_check_views = [5, 6, 4, 6, 5, 5, 6, 5]

