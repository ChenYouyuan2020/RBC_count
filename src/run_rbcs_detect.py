import cv2
import numpy as np
import sys 
from utils import MyFluoImgs, MyFluoRatio, preprocess_imgs
from matplotlib import pyplot as plt
import os
import argparse
import pandas as pd 
import glob
import nd2
import logging
import argparse 

parser = argparse.ArgumentParser(description="rbc_intensity_ratio")
parser.add_argument("--read_dir", type=str)
parser.add_argument("--save_dir", type=str,default=None)
parser.add_argument("--log_dir", type=str,default=None)
parser.add_argument("--if_cut", type=bool, default=False)
args = parser.parse_args()

def rbc_count(folder_path, save_dir, if_cut=False):
    """
    Args:
        folder_path (str): folder path containing .nd2 files.
        save_dir (str): folder path to save results.
        if_cut (bool): whether to cut the images to a smaller size.
    
    Returns:
        None
    """

    image_files = glob.glob(os.path.join(folder_path, "*.nd2")) 
    try:
        assert len(image_files) ==1 , "只能有一个 .nd2"

        # Load images into lists
        nd2_file = nd2.ND2File(image_files[0])
        images = nd2_file.asarray()*16
        nd2_file.close() 

        #DAPI blue RGB 2 1230
        #FITC green RGB 1 1240
        if if_cut:
            images = preprocess_imgs(images, cut_range_x=[100,924], cut_range_y=[100,924])

        images_1230 = images[:, 0, :, :]
        images_1240 = images[:, 1, :, :]
        images_merge = np.mean(images[:, 0:2, :, :],axis=1)

        min_radius = 4 #5 # this is the min radius of the RBCs, it should be smaller than 3 to remove the noise (可以只有有部分细胞，然后找补)
        max_radius = 18*4 #15 # this is the max radius of the RBCs, it should be larger than 10 to remove the noise （可以是多个细胞的集合，然后通过圆检查分裂）
        filter_sigma= 10 # this is the sigma of the gaussian filter, it is not very sensitive, but it should be larger than 10 to remove the illumination
        Otsu_thresh = 220 # this has to be adjusted for each image, but it is not very sensitive
        circularity_range = 1 # the range of circularity to be considered as a RBC
        fill_size = 4 # 2 pixels is experimentally the best to exclude connected RBCs and not fill the holes in the RBCs

        background_max = 0.05 * 2**16 # 筛除背景离群点
        signal_max = 0.70 * 2**16  # 筛除信号离群点

        ImgMerge = MyFluoImgs(images_merge,'merge',save_dir=save_dir)
        ImgMerge.describe_image(min_radius, max_radius, 
                                filter_sigma, Otsu_thresh, 
                                circularity_range, fill_size, 
                                signal_max,background_max,)

        ImgMerge.plot_all()
        ImgMerge.hough_circles(dp=1.1, minDist=16, param1=50, param2=9, minRadius=8, maxRadius=15)
        best_layer = ImgMerge.get_best_layer()

        # 设置 intensity ratio
        FlouRatio = MyFluoRatio(images_1230, images_1240, ImgMerge, save_dir=save_dir)
        FlouRatio.set_larger_outlines_mask(outlines_mask_ratio = 1.3)
        df = FlouRatio.get_ratio_1230vs1240()
        
        if len(df)>=1:
            df.to_csv(os.path.join(save_dir, 'ratio_1230vs1240.csv'))

        FlouRatio.sc_otsu_intensity_plot(images_merge[best_layer], intensity_keys='intensity_1230', reserved_keys='SingleCell_reserved_1230', title='__4_intensity_1230____SingleCell_reserved_1230.png')
        FlouRatio.sc_otsu_intensity_plot(images_merge[best_layer], intensity_keys='intensity_1240', reserved_keys='SingleCell_reserved_1240', title='__4_intensity_1240____SingleCell_reserved_1240.png')
        FlouRatio.sc_otsu_intensity_plot(images_merge[best_layer], intensity_keys='intensity_1230', reserved_keys='SingleCell_reserved_30_40', title='__4_intensity_1230____SingleCell_reserved_30_40.png')
        FlouRatio.sc_otsu_intensity_plot(images_merge[best_layer], intensity_keys='intensity_1240', reserved_keys='SingleCell_reserved_30_40', title='__4_intensity_1240____SingleCell_reserved_30_40.png')

        FlouRatio.intensity_with_mask_plot(title='__5_intensity_with_mask.png')
        
        logging.info(
                    f"{os.path.basename(save_dir)} SingleCell_reserved_30_40 with ratio "
                    f"{len(df[df['SingleCell_reserved_30_40']])/len(df):.2%} pass filtered and have "
                    f"{np.max(df['cell'].values)} cells !!!"
                    )
        
    except AssertionError as ae:
        logging.error(f'Error in {os.path.basename(save_dir)} with {len(image_files)}')


if __name__ == '__main__':

    read_dir = args.read_dir
    save_dir = args.save_dir
    log_dir = args.log_dir
    if_cut = args.if_cut
    
    logging.getLogger('matplotlib').setLevel(logging.ERROR)
    logging.basicConfig(filename=os.path.join(log_dir,"rbc.log"), level=logging.INFO, format='%(asctime)s %(message)s')
    logging.info(f"read_dir: {read_dir}")
    logging.info(f"save_dir: {save_dir}")
    logging.info(f"log_dir: {log_dir}")
    logging.info("  ")

    rbc_count(read_dir, save_dir, if_cut=False)



