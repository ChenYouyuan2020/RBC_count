import enum
import cv2
import numpy as np
from skimage import filters, morphology, measure
from skimage.morphology import disk
from scipy.ndimage import binary_fill_holes
import matplotlib.pyplot as plt
import seaborn as sns 
from scipy.stats import linregress
import os
import pandas as pd 
import seaborn as sns
import logging

def circular_top_hat(image, radius):
    # 创建圆形的结构元素，使用skimage的disk函数来生成
    selem = morphology.disk(radius)
    
    # 进行开运算（腐蚀再膨胀），获取底部区域
    opened = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, selem)

    # result = cv2.subtract(image, opened)
    return opened

def correct_illumination(image, sigma=50):
    """
    使用高斯模糊校正光照不均匀
    """
    blurred = cv2.GaussianBlur(image, (0, 0), sigma)
    corrected = cv2.addWeighted(image, 1, blurred, -1, 128)
    # corrected = image - blurred*0.95
    return corrected, blurred

def detect_circles(image, min_radius=10, max_radius=100):
    """
    使用Hough圆变换检测圆形区域
    """
    image = image.astype(np.uint8)
    # Hough圆变换
    circles = cv2.HoughCircles(
        image, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
        param1=50, param2=30, minRadius=min_radius, maxRadius=max_radius
    )
    return circles

def create_circular_mask(image_shape, circles):
    """
    根据检测到的圆形创建掩码
    """
    mask = np.zeros(image_shape, dtype=np.uint8)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for circle in circles[0, :]:
            center = (circle[0], circle[1])
            radius = circle[2]
            cv2.circle(mask, center, radius, 1, -1)  # 填充圆形
    return mask

def divide_into_blocks(image, block_size):
    """
    将图像分割为 block_size x block_size 的块
    """
    h, w = image.shape
    blocks = []
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = image[i:i+block_size, j:j+block_size]
            blocks.append(block)
    return blocks

def local_otsu_thresholding(blocks):
    """
    对每个块应用局部 Otsu 阈值分割
    """
    binary_blocks = []
    for block in blocks:
        if block.size > 0:  # 确保块不为空
            thresh = filters.threshold_otsu(block)
            binary_block = block > thresh
            binary_blocks.append(binary_block)
    return binary_blocks

def merge_blocks(binary_blocks, image_shape, block_size):
    """
    将分割后的块合并为完整的二值图像
    """
    h, w = image_shape
    merged_image = np.zeros((h, w), dtype=np.uint8)
    index = 0
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            if index < len(binary_blocks):
                merged_image[i:i+block_size, j:j+block_size] = binary_blocks[index]
                index += 1
    return merged_image

def extract_non_connected_circles(image, min_radius=10, max_radius=100, filter_sigma=20, Otsu_thresh=60, circularity_range=0.2, fill_size = 9):
    """
    提取图像中非连通的圆形区域
    """
    # 校正光照不均匀
    # corrected, blurred = correct_illumination(image, filter_sigma)
    # corrected = (corrected/corrected.max()*255).astype(np.uint8)
    # binary_image = (corrected >= Otsu_thresh)

    image = cv2.normalize(image.copy(), None, 0, 255, cv2.NORM_MINMAX,dtype=cv2.CV_8U)
    blurred = cv2.medianBlur(image, 3)
    _, thresh = cv2.threshold(blurred, 10, 255,  cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    #binary_image = (corrected >= Otsu_thresh)
    binary_image = 255-thresh
   
    # # 后处理：形态学操作
    binary_image = binary_fill_holes(binary_image)  # 填充空洞
    # se_clear = disk(2)  # 定义结构元素
    se_fill = disk(fill_size)
    # binary_image = morphology.opening(binary_image, se_clear)  # 开运算去除噪声
    binary_image = morphology.closing(binary_image, se_fill)  # 闭运算填充空洞
    
    # 提取非连通区域
    labeled_image = measure.label(binary_image)
    regions = measure.regionprops(labeled_image)
    cell_num = 0
    
    # 过滤掉过小或过大的区域，并判断形状是否接近圆形
    final_mask = np.zeros_like(binary_image, dtype=np.uint8)
    for region in regions:
        area = region.area
        perimeter = region.perimeter
        if perimeter > 0:
            circularity = 4 * np.pi * (area / (perimeter ** 2))  # 圆形度计算公式
            if min_radius**2 * np.pi <= area <= max_radius**2 * np.pi and 1-circularity_range <= circularity <= 1+circularity_range:
                final_mask[labeled_image == region.label] = 1
                cell_num += 1
    final_mask = morphology.opening(final_mask, se_fill)  # 
    
    return blurred, binary_image, final_mask, cell_num

def corr_matrix(images_1230,images_1240, best_1230,dir):
    corr_matrix = np.zeros((len(images_1230), len(images_1240))) # 通过corrmatrix计算1230和1240的偏移量
    for i, img1 in enumerate(images_1230):
        img1_flat = img1.flatten()
        for j, img2 in enumerate(images_1240):
            img2_flat = img2.flatten()
            # Compute Pearson correlation
            if img1_flat.std() > 0 and img2_flat.std() > 0:
                corr = np.corrcoef(img1_flat, img2_flat)[0, 1]
            else:
                corr = 0
            corr_matrix[i, j] = corr

    shift = np.argmax(corr_matrix,axis=1)
    x = np.arange(len(images_1230))
    y = shift
    res = linregress(x, y)
    slope, intercept = res.slope, res.intercept

    figs,axes = plt.subplots(1,2,figsize=(6,2))
    sns.heatmap(corr_matrix,ax=axes[0])
    axes[0].set_xlabel("1240")
    axes[0].set_ylabel("1230")

    axes[1].scatter(x,y,color='blue')
    axes[1].plot(x, slope*x + intercept, color='green', label='fit line')
    axes[1].scatter([best_1230],[slope*best_1230 + intercept],color='red')
    axes[1].set_xlabel("1230")
    axes[1].set_ylabel("1240")
    plt.title(f"1230:{best_1230} vs 1240:{int(slope*best_1230 + intercept)}")
    plt.tight_layout()
    plt.show()
    plt.savefig(os.path.join(dir,"__2_corrationMatirx.png"))
    plt.close('all')

    return int(slope*best_1230 + intercept)


# 定义一个函数计算两点之间的欧几里得距离
def euclidean_distance(point1, point2):
    return np.sqrt(np.abs((np.int64(point1[0]) - np.int64(point2[0]))**2 
                        + (np.int64(point1[1]) - np.int64(point2[1]))**2))

def describe_image(images, title, 
                   min_radius, max_radius, 
                   filter_sigma ,Otsu_thresh, 
                   circularity_range, fill_size,
                   signal_max, background_max,save_dir):
    
    corrected_images, binary_images,final_masks = [], [], []
    describes = {"group":[], "intensity":[], "intensity_type":[]}
    cells = []

    for i,image in enumerate(images):
        corrected_image,  binary_image, final_mask,cell = \
            extract_non_connected_circles(image, min_radius, max_radius, filter_sigma ,Otsu_thresh, circularity_range, fill_size)

        cut_index_x = image.shape[0]//3
        cut_index_y = image.shape[1]//3

        corrected_images.append(corrected_image)
        binary_images.append(binary_image)
        final_masks.append(final_mask)
        
        cut_mask = binary_image[cut_index_x:-cut_index_x, cut_index_y:-cut_index_y]
        cut_image = image[cut_index_x:-cut_index_x, cut_index_y:-cut_index_y]

        signal = cut_image[cut_mask == 1]
        background = cut_image[~(cut_mask==1)]

        """sum_signal = np.sum(cut_image[cut_mask == 1])
        sum_background = np.sum(cut_image[cut_mask == 0])
        sum_signal_count = np.sum(cut_mask == 1) if np.sum(cut_mask == 1) !=0 else 1e-20
        sum_background_count = np.sum(cut_mask == 0) if np.sum(cut_mask == 0) !=0 else 1e-20

        signal_mean = sum_signal / sum_signal_count
        background_mean = sum_background / sum_background_count"""

        describes['intensity'].extend(signal)
        describes['intensity_type'].extend(['signal']*len(signal))
        describes['intensity'].extend(background)
        describes['intensity_type'].extend(['background']*len(background))
        describes['group'].extend([i]*(len(signal)+len(background)))
        cells.append(cell)

    describes = pd.DataFrame(describes)

    threshold = background_max # 背景强度不能太高
    mask = (describes['intensity_type'] == 'background') & (describes['intensity'] >= threshold)
    describes = describes[~mask]

    threshold = signal_max # 信号不能太强
    mask = (describes['intensity_type'] == 'signal') & (describes['intensity'] >= threshold)
    describes = describes[~mask]

    fig, axes = plt.subplots(1, 4, figsize=(15,3))  # 返回fig对象和axes数组
    signal_means_1    = intensity_plot(describes, intensity_type='signal', ax=axes[0])
    background_means_1  = intensity_plot(describes, intensity_type='background', ax=axes[1])

    axes[2].plot(cells)
    axes[2].set_title(f'Cell Count: {np.argmax(cells)}')

    SBRs = [i/j if j != 0 else 0 for i,j in zip(signal_means_1, background_means_1)]
    SBRs = [i if j>15 else 0 for i,j in zip(SBRs, cells)]
    axes[3].plot(SBRs)
    axes[3].set_title(f'SBR :, {np.argmax(SBRs)}')

    plt.suptitle(f"{title}: Use SBRs {np.argmax(SBRs)} ") 
    plt.tight_layout()
    if save_dir: 
        plt.savefig(os.path.join(save_dir,f'__1__Describe_{title}.png'))
        plt.close('all')

    #logging.info(f"{os.path.basename(save_dir)} have {cells[np.argmax(SBRs)]} cells !!!")
    #print(f"{os.path.basename(save_dir)} have {cells[np.argmax(SBRs)]} cells !!!")

    return corrected_images, binary_images, final_masks, np.argmax(SBRs)

def corr_matrix(images_1230,images_1240, best_1230):
    corr_matrix = np.zeros((len(images_1230), len(images_1240))) # 通过corrmatrix计算1230和1240的偏移量
    for i, img1 in enumerate(images_1230):
        img1_flat = img1.flatten()
        for j, img2 in enumerate(images_1240):
            img2_flat = img2.flatten()
            # Compute Pearson correlation
            if img1_flat.std() > 0 and img2_flat.std() > 0:
                corr = np.corrcoef(img1_flat, img2_flat)[0, 1]
            else:
                corr = 0
            corr_matrix[i, j] = corr

    shift = np.argmax(corr_matrix,axis=1)
    x = np.arange(len(images_1230))
    y = shift
    res = linregress(x, y)
    slope, intercept = res.slope, res.intercept

    figs,axes = plt.subplots(1,2,figsize=(6,2))
    sns.heatmap(corr_matrix,ax=axes[0])
    axes[0].set_xlabel("1240")
    axes[0].set_ylabel("1230")

    axes[1].scatter(x,y,color='blue')
    axes[1].plot(x, slope*x + intercept, color='green', label='fit line')
    axes[1].scatter([best_1230],[slope*best_1230 + intercept],color='red')
    axes[1].set_ylabel("1230")
    axes[1].set_xlabel("1240")
    plt.title(f"1230:{best_1230} vs 1240:{int(slope*best_1230 + intercept)}")
    plt.tight_layout()
    plt.show()

    return int(slope*best_1230 + intercept)

def intensity_plot(data, intensity_type, ax, **args):
    data = data[data['intensity_type'] == intensity_type]
    sns.boxplot(x='group', y='intensity', data=data, ax=ax)
    sns.stripplot(x='group',y='intensity',data=data, ax=ax, color='black', size=0.1, **args)
    means = data.groupby('group')['intensity'].mean()
    x_order = data['group'].unique()
    means_1 = [means[group] for group in x_order]
    ax.plot(range(len(x_order)), means, color='red', marker='o', linewidth=2, label='Mean')
    ax.set_title(f'{intensity_type} : {np.argmax(means)}')

    # 计算离群点
    q1 = data['intensity'].quantile(0.25)
    q3 = data['intensity'].quantile(0.75)
    iqr = q3 - q1
    initers = data[(data['intensity'] >= q1 - 1.5 * iqr) & (data['intensity'] <= q3 + 1.5 * iqr)]
    means = initers.groupby('group')['intensity'].mean()
    x_order = data['group'].unique()
    # means_2 = [means[int(group)] for group in x_order]
    #ax.plot(range(len(x_order)), means, color='green', marker='o', linewidth=2, label='Mean (no outliers)')
    return means_1 # means_2

class MyFluoImgs():
    def __init__(self, images, title,save_dir=None):
        self.images = images
        self.binary_imgs = None
        self.corrected_imgs = None
        self.final_masks = None
        self.best = None
        self.title = title
        self.mask = None
        self.save_dir = save_dir

    def describe_image(self,
                    min_radius, max_radius, filter_sigma, Otsu_thresh, 
                    circularity_range, fill_size,
                    signal_max, background_max):
        
        self.corrected_imgs,self.binary_imgs,self.final_masks, self.best  = \
        describe_image(self.images,self.title,
                        min_radius, max_radius, filter_sigma, Otsu_thresh, 
                        circularity_range, fill_size,
                        signal_max, background_max,self.save_dir)
        
    def plot_all(self):
        plt.figure(figsize=(8, 8))
        plt.subplot(221)
        plt.imshow(self.images[self.best])
        plt.title("Original")
        plt.subplot(222)
        plt.imshow(self.corrected_imgs[self.best])
        plt.title("Corrected")
        plt.subplot(223)
        plt.imshow(self.binary_imgs[self.best])
        plt.title("Initial Mask (forCheck)")
        plt.subplot(224)
        plt.imshow(self.final_masks[self.best])
        plt.title("Result (forUse)")
        plt.suptitle(f"Image {self.title}")
        if self.save_dir: 
            plt.savefig(os.path.join(self.save_dir,f'__2__CellCount_{self.title}.png'))
            plt.close('all')   
     
    # 使用 HoughCircles 检测圆圈, 分离相邻细胞
    def hough_circles(self,dp, minDist, param1, param2, minRadius, maxRadius):
        """
        - `image`：输入的灰度图像，建议先用 `cv2.cvtColor` 转为灰度。
        - `method`：检测方法，通常用 `cv2.HOUGH_GRADIENT`。
        - `dp`：累加器分辨率与原图分辨率的反比，如 `dp=1` 表示一致，`dp=2` 为原图一半。
        - `minDist`：检测到的圆心之间的最小距离（像素），防止检测到多个相近的圆。
        - `param1`: Canny 边缘检测的高阈值，影响边缘检测灵敏度。
        - `param2`：圆心累加器的阈值，越小越灵敏（易误检），越大则只检测明显的圆。
        - `minRadius`：检测圆的最小半径（像素）。
        - `maxRadius`：检测圆的最大半径（像素）。
        """
        initial_images = self.final_masks[self.best].copy().astype('uint8')
        initial_images = cv2.normalize(initial_images, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        circles = cv2.HoughCircles(initial_images, cv2.HOUGH_GRADIENT, dp=dp, minDist=minDist,
                        param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)
        
        img_with_circles = cv2.cvtColor(initial_images, cv2.COLOR_GRAY2BGR)  # 转换为 BGR 格式以便绘制彩色圆

        # 如果检测到圆
        if circles is not None:
            circles = np.uint16(np.around(circles))  # 四舍五入并转换为 uint16
            for i in circles[0, :]:
                # 绘制圆的轮廓
                cv2.circle(img_with_circles, (i[0], i[1]), i[2], (0, 255, 0), 2)
        try:
            self.circles = circles[0]
        except:
            self.circles = [[0,0,0]]
        self.mask = self.circle_mask()

        plt.figure(figsize=(12, 4))
        plt.subplot(131)
        plt.imshow(initial_images)
        plt.title("initial_images")
        plt.subplot(132)
        plt.imshow(img_with_circles)
        plt.title("Detected Circles in initial_images")
        plt.subplot(133)
        plt.imshow(self.mask)
        plt.title("Mask")
        plt.suptitle(" HoughCircles ")
        if self.save_dir: 
            plt.savefig(os.path.join(self.save_dir,f'__3__hough_circles_{self.title}.png'))
            plt.close('all')
        
    def circle_mask(self):
        mask = np.zeros(self.images[self.best].shape[:2], dtype=np.int32)
        for i, (x, y, radius) in enumerate(self.circles):
            x, y, radius = map(int, [x, y, radius])
            yy, xx = np.ogrid[:mask.shape[0], :mask.shape[1]]
            dist = (xx - x) ** 2 + (yy - y) ** 2
            mask[dist <= radius ** 2] = i + 1
        return mask
    def get_outline_larger_mask(self, larger_ratio=1.3):
        if self.mask is None:
            print("Please run the hough_circles method first")
        mask = np.zeros(self.images[self.best].shape[:2], dtype=np.int32)
        for i, (x, y, radius) in enumerate(self.circles):
            x, y, radius = map(int, [x, y, radius])
            radius = radius*larger_ratio
            yy, xx = np.ogrid[:mask.shape[0], :mask.shape[1]]
            dist = (xx - x) ** 2 + (yy - y) ** 2
            mask[dist <= radius ** 2] = i + 1
        return mask

    def get_mask(self):
        if self.mask is None:
            print("Please run the hough_circles method first")
        return self.mask
    
    def get_best_layer(self):
        if self.best is None:
            print("Please run the describe_image method first")
        return self.best

    def get_images(self):
        return self.images




    