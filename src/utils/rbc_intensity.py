import numpy as np 
import pandas as pd
from .rbc_count import MyFluoImgs 
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os 

def rbc_intensiy(img, mask):
    intensity = img.flatten()
    cells = mask.flatten()
    img_index = np.arange(len(cells))

    # 创建一个数据框，包含图像的强度、细胞和图像索引
    df = pd.DataFrame({'intensity': intensity, 'cell': cells,'img_index': img_index})
    # df = df[df['cell'] != 0]
    df.loc[df['cell'] == 0, 'intensity'] = 0
    return df

def Otsu_intensity_16bit(intensity):
    data_16bit = np.array(intensity, dtype=np.uint16)
    data_sorted = np.sort(data_16bit)  # 按照从小到大排序（方便分割）

    max_var = -1
    best_t = None

    unique_values = np.unique(data_sorted)   # 获得所有唯一的可能阈值

    for t in unique_values[:-1]:    # 不用最后一个，因为分成空集
        mask0 = data_sorted <= t
        mask1 = data_sorted > t
        w0 = np.mean(mask0)   # 等价于 len(mask0_true)/len(data)
        w1 = np.mean(mask1)
        if w0 == 0 or w1 == 0:
            continue
        mu0 = np.mean(data_sorted[mask0])
        mu1 = np.mean(data_sorted[mask1])
        var_between = w0 * w1 * (mu0 - mu1) ** 2
        if var_between > max_var:
            max_var = var_between
            best_t = t

    return data_16bit>=best_t if best_t is not None else False

def sc_otsu_intensity_plot(img, cell_df, intensity_keys='intensity', reserved_keys='is_fg',save_dir=None,title='images.png'):
    h,w = img.shape
    fig = plt.figure(figsize=(12,8))
    gs = gridspec.GridSpec(2, 4, height_ratios=[1, 2])  # 2行4列，比例1:2

    # ========================
    # 第一行左侧1/4：cell==10 mask
    # ax_mask10 = fig.add_subplot(gs[0, 0])
    # cell_10 = cell_df.copy()
    # cell_10[reserved_keys] = 1    
    # cell_10.loc[cell_10['cell'] == 10, reserved_keys] = 2
    # cell_10.loc[cell_10['cell'] == 0, reserved_keys] = 0
    # ax_mask10.imshow(cell_10[reserved_keys].values.reshape(h, w), cmap='plasma')
    # ax_mask10.set_title('cell==10 mask')
    # ax_mask10.axis('off')

    # 第一行右侧3/4：直方图，跨三列
    ax_hist = fig.add_subplot(gs[0, :])
    cell_10 = cell_df.copy()
    cell_10 = cell_10[cell_10[intensity_keys]>250]

    notF_list = cell_10[cell_10[reserved_keys]][intensity_keys].values
    hasF_list = cell_10[~cell_10[reserved_keys]][intensity_keys].values
    notF_list  = [x for x in notF_list if x != np.inf]
    hasF_list  = [x for x in hasF_list if x != np.inf]

    ax_hist.hist(notF_list, bins=100, color='orange', alpha=0.6, label='not_filtered')
    ax_hist.hist(hasF_list, bins=100, color='dodgerblue', alpha=0.6, label='has filtered')
    ax_hist.set_title('all signle cell otsu intensity cutoff')
    ax_hist.legend()

    # ========================
    # 第二行左侧：原图
    iimag = cell_df[intensity_keys].values.reshape(h, w)
    iimag[np.isinf(iimag)] = 0
    ax_img = fig.add_subplot(gs[1, 0:2])
    ax_img.imshow(iimag, cmap='hot')
    ax_img.set_title('original image')
    ax_img.axis('off')

    # 第二行右侧：otsu mask图
    ax_mask = fig.add_subplot(gs[1, 2:])
    ax_mask.imshow(cell_df[reserved_keys].values.reshape(h, w), cmap='plasma')
    ax_mask.set_title('single cells otsu masked image')
    ax_mask.axis('off')

    plt.suptitle(f"{intensity_keys}    -    {reserved_keys}")
    plt.tight_layout(h_pad=2, w_pad=2)
    plt.show()
    if save_dir:
        plt.savefig(os.path.join(save_dir, title))
        plt.close('all')

def otsu_single_cell_mask(img, outlines_larger_mask):
    cell_df = rbc_intensiy(img, outlines_larger_mask)
    cell_df['is_fg'] = cell_df.groupby('cell')['intensity'].transform(Otsu_intensity_16bit)
    #cell_df = cell_df[cell_df['intensity']!=0]
    return cell_df

class MyFluoRatio():
    def __init__(self, imgs_1230, imgs_1240, myfluoimg_merge: MyFluoImgs,save_dir):
        self.imgs_1230 = imgs_1230
        self.imgs_1240 = imgs_1240
        self.myfluoimg_merge = myfluoimg_merge
        self.mask = myfluoimg_merge.get_mask()
        self.mask_larger_outlines = None
        self.best = myfluoimg_merge.get_best_layer()
        self.otsu_1230_sc_mask = None
        self.otsu_1240_sc_mask = None
        self.otsu_merge_sc_mask = None
        self.df_sc_img = None  # 每个像素保留了多条single cells 信息的图片
        self.save_dir = save_dir
        

    def set_larger_outlines_mask(self, outlines_mask_ratio= 1.3):
        self.mask_larger_outlines = self.myfluoimg_merge.get_outline_larger_mask(outlines_mask_ratio)

    def get_ratio_1230vs1240(self):
        assert self.mask_larger_outlines is not None, "please run <set_larger_outlines_mask> first"
        df_1230 = otsu_single_cell_mask(self.imgs_1230[self.best], self.mask_larger_outlines)
        df_1240 = otsu_single_cell_mask(self.imgs_1240[self.best], self.mask_larger_outlines)
        df_merge = otsu_single_cell_mask(self.myfluoimg_merge.get_images()[self.best], self.mask_larger_outlines)

        df_1230.rename(columns={'intensity': 'intensity_1230',"is_fg":"SingleCell_reserved_1230"},inplace=True)
        df_1240.rename(columns={'intensity': 'intensity_1240',"is_fg":"SingleCell_reserved_1240"},inplace=True)
        df_merge.rename(columns={'intensity': 'intensity_merge',"is_fg":"SingleCell_reserved_merge"},inplace=True)

        # 新的图片信号
        new_df = df_merge
        new_df['intensity_1230'] = df_1230['intensity_1230']
        new_df['intensity_1240'] = df_1240['intensity_1240']
       
        new_df['SingleCell_reserved_1230'] = df_1230['SingleCell_reserved_1230']
        new_df['SingleCell_reserved_1240'] = df_1240['SingleCell_reserved_1240']

        new_df['intensity_ratio_1230vs1240'] = new_df['intensity_1230'] / new_df['intensity_1240']

        # create SingleCell_reserved_30_40_merge  1230&1240&merge both ture
        # create SingleCell_reserved_30_40 1230&1240 both ture
        new_df['SingleCell_reserved_30_40_merge'] = new_df['SingleCell_reserved_1230'] & new_df['SingleCell_reserved_1240'] & new_df['SingleCell_reserved_merge']
        new_df['SingleCell_reserved_30_40'] = new_df['SingleCell_reserved_1230'] & new_df['SingleCell_reserved_1240']
        
        self.df_sc_img = new_df.copy()

        # 删除冗余
        new_df = new_df[new_df['cell'] != 0]
        new_df = new_df[new_df['intensity_1230']!=0]
        new_df = new_df[new_df['intensity_1240']!=0]

        # 添加最高层
        new_df['best'] = self.best
        self.new_df = new_df

        return new_df
        
    # 定义一个函数，用于绘制sc-otsu强度图
    def sc_otsu_intensity_plot(self, img, intensity_keys='intensity', reserved_keys='is_fg',title="image.png"):
        # 调用sc_otsu_intensity_plot函数，传入img、self.df_sc_img、intensity_keys和reserved_keys参数
        sc_otsu_intensity_plot(img, self.df_sc_img, intensity_keys, reserved_keys, self.save_dir, title=title)

    def intensity_with_mask_plot(self,title="image.png"):
        plt.figure(figsize=(12,7))
        plt.subplot(231)
        ratio = self.new_df['intensity_ratio_1230vs1240'].values
        ratio = np.log10(ratio)
        _ = plt.hist(ratio, bins=100)
        plt.title("intensity_ratio_1230vs1240")

        plt.subplot(232)
        ratio = self.new_df[self.new_df['SingleCell_reserved_30_40']]['intensity_ratio_1230vs1240'].values
        ratio = np.log10(ratio)
        _ = plt.hist(ratio, bins=100)
        plt.title("SingleCell_reserved_30_40")

        plt.subplot(233)
        ratio = self.new_df[self.new_df['SingleCell_reserved_1230']]['intensity_ratio_1230vs1240'].values
        ratio = np.log10(ratio)
        _ = plt.hist(ratio, bins=100)
        plt.title("SingleCell_reserved_1230")

        plt.subplot(234)
        ratio = self.new_df[self.new_df['SingleCell_reserved_1240']]['intensity_ratio_1230vs1240'].values
        ratio = np.log10(ratio)
        _ = plt.hist(ratio, bins=100)
        plt.title("SingleCell_reserved_1240")

        plt.subplot(235)
        ratio = self.new_df[self.new_df['SingleCell_reserved_merge']]['intensity_ratio_1230vs1240'].values
        ratio = np.log10(ratio)
        _ = plt.hist(ratio, bins=100)
        plt.title("SingleCell_reserved_merge")

        plt.subplot(236)
        ratio = self.new_df[self.new_df['SingleCell_reserved_30_40_merge']]['intensity_ratio_1230vs1240'].values
        ratio = np.log10(ratio)
        _ = plt.hist(ratio, bins=100)
        plt.title("SingleCell_reserved_30_40_merge")
        plt.show()

        if self.save_dir:
            plt.savefig(os.path.join(self.save_dir, title))
            plt.close('all')




        


       



    