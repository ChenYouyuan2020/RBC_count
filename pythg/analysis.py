import cv2
import numpy as np
import matplotlib.pyplot as plt

def read_plot(path_biotin):
    fluo = cv2.imread(path_biotin,3)
    fluo= cv2.cvtColor(fluo, cv2.COLOR_BGR2GRAY)
    #fluo = cv2.normalize(fluo,None,0,255,cv2.NORM_MINMAX)
    plt.figure()
    plt.imshow(fluo,cmap='hot')
    #plt.imshow(fluo)
    plt.title(f"imging of fluo deepth:{np.max(fluo)}")
    plt.axis('off')
    plt.show()
    print(fluo.shape)
    return fluo

def chose_mask(fluo, min_area=5, max_area=7,min_degree=0.6, radius_hist_bin=50,erode_iterations=0):
    #ostu 二值化
    _, fluo_binary = cv2.threshold(fluo, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    fluo_binary = fluo_binary.astype(np.uint8)
     #erode
    if erode_iterations != 0:
        kernel = np.ones((3,1),np.uint8)
        fluo_binary = cv2.erode(fluo_binary,kernel,iterations=erode_iterations)
        fluo_binary = cv2.dilate(fluo_binary, kernel, iterations = erode_iterations-1)
    contours, hierarchy = cv2.findContours(fluo_binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    #contours, hierarchy = cv2.findContours(fluo_binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    staticstic_mask(contours, hierarchy, radius_hist_bin,suffix='original')
    new_image2,index_list, area_list,arc_list,degree_list = \
        choose_mask(fluo_binary, contours, hierarchy, min_area, max_area,min_degree, radius_hist_bin)
    
    new_contours = [contours[i] for i in index_list]
    new_hierarchy = [[hierarchy[0][i] for i in index_list]]

    staticstic_mask(new_contours,new_hierarchy, radius_hist_bin,suffix='filtered')
    
    plt.figure(figsize=(30,10))
    plt.subplot(133)
    plt.title("The totall number of cells %d" % len(new_contours))
    plt.imshow(new_image2)

    plt.subplot(131)
    plt.imshow(fluo,cmap='hot')
    plt.title("Original image")

    plt.subplot(132)
    plt.imshow(fluo_binary,cmap='hot')
    plt.title("Binary image")

    return  contours, hierarchy, index_list, area_list, arc_list, degree_list,fluo_binary

def staticstic_mask(contours, hierarchy, radius_hist_bin=50,suffix='original'):
    area_list = []
    arc_list = []
    degree_list = []

    for i,cnt in enumerate(contours):
        if hierarchy[0][i][-1]==-1:
            area = cv2.contourArea(cnt)
            arc = cv2.arcLength(cnt,1)
            if arc > 1e-5:
                degree = 4*np.pi*area/(arc*arc)
                area_list.append(area)
                arc_list.append(arc)
                degree_list.append(degree)
                
    plt.figure(figsize=(30,6))       
    plt.subplot(131)
    _ = plt.hist(area_list, bins=radius_hist_bin)
    plt.title(f"{suffix} area_list of the cells")
    
    plt.subplot(132)
    _ = plt.hist(arc_list,bins=radius_hist_bin)
    plt.title(f"{suffix} arc_list of the cells")
    
    plt.subplot(133)
    _ = plt.hist(degree_list,bins=radius_hist_bin)
    plt.title(f"{suffix} degree_list of the cells")
    
def choose_mask(fluo, contours, hierarchy, min_area=5, max_area=7,min_degree=0.6,radius_hist_bin=50):
    new_image2 = np.zeros((fluo.shape[0],fluo.shape[1],3),dtype=np.uint16)
    new_image2[:,:,2] = fluo
    new_image2[:,:,1] = fluo
    new_image2 = new_image2.astype(np.uint16)

    index_list = []
    area_list = []
    arc_list = []
    degree_list = []

    for i,cnt in enumerate(contours):
        if hierarchy[0][i][-1]==-1:
            mask=np.zeros_like(fluo)
            cv2.drawContours(mask,contours, i, 1, cv2.FILLED)
            area = np.sum(mask)
            arc = cv2.arcLength(cnt,1)
            degree = 0               
            if arc > 1e-5:
                degree = 4*np.pi*area/(arc*arc)
 
            if area> min_area and area < max_area:
                if degree > min_degree and arc>0:
                    cv2.drawContours(new_image2 ,contours,i,(255,0,0),2)
                    index_list.append(i)
                    area_list.append(area)
                    arc_list.append(arc)
                    degree_list.append(degree)
                    
    return new_image2,index_list, area_list,arc_list,degree_list

def plot_mask(fluo, fluo_binary,contours,index_list,hole_ratio=0.2):
    new_image2 = np.zeros((fluo.shape[0],fluo.shape[1],3),dtype='int')
    #new_image2[:,:,2] = fluo
    new_image2[:,:,-1] = fluo_binary
    new_image2 = new_image2.astype(np.uint8)

    new_contours = [contours[i] for i in index_list]

    intensities = []
    x_list = []
    y_list = []
    radius_list = []

    for i,cnt in enumerate(new_contours):
        mask=np.zeros_like(fluo)
        mask1 = np.zeros_like(fluo)
        (x,y),radius = cv2.minEnclosingCircle(cnt)
        cv2.drawContours(mask,new_contours, i, 1, cv2.FILLED)
        
        #if new_hierarchy[0][i][-2] != -1:
        mask1[(mask==1)&(fluo_binary==0)] = 1 # the hole area is mask1, 
        count = np.sum(mask)
        count1 = np.sum(mask1)
        if count1/count > hole_ratio:
            intensity = 0
            new_image2[(mask==1)&(fluo_binary==0)] = [0,255,0]
        else:
            cv2.drawContours(new_image2, new_contours, i, (255,0,0), 2)
            mask[(mask==1)&(fluo_binary==0)] = 0
            intensity = np.mean(fluo[mask==1])
            
        intensities.append(intensity)
        x_list.append(x)
        y_list.append(y)
        radius_list.append(radius)
       
    plt.figure()
    num = np.count_nonzero(np.array(intensities))
    plt.title(f"removed cells {num} (green) \n preserved cells {len(intensities)-num} (red)")
    plt.imshow(new_image2)
    
    
    return intensities, x_list, y_list, radius_list

def save_file(flod, fluo_name,index_list,contours, intensities, area_list, arc_list, x_list, y_list, radius_list,bins=30):
    plt.figure()
    _ = plt.hist(intensities, bins)
    plt.title('hist of intensities')
    intensities = np.array(intensities).reshape(-1,1)
    area_list = np.array(area_list).reshape(-1,1)
    arc_list = np.array(arc_list).reshape(-1,1)
    x_list = np.array(x_list).reshape(-1,1)
    y_list = np.array(y_list).reshape(-1,1)
    radius_list = np.array(radius_list).reshape(-1,1)
   
    out_file = np.concatenate([x_list,y_list,radius_list,intensities, area_list, arc_list],axis=1)
    
    out_file = out_file[out_file[:,3] !=0]
    
    head = r"x,y,radius,intensities,area,arc"
    print(out_file.shape[0], "file has been saved")
    print(len(area_list),"real cell numbers")
    np.savetxt(flod+'\\'+fluo_name+'.csv',out_file,fmt='%.4f',delimiter=',',header=head)
    
    """
    file_path = os.path.join(flod,fluo_name+'.txt')
    new_contours = [contours[i] for i in index_list]
    with open(file_path,'w') as fo:
        for i in new_contours:
            for x in i:
                for y in x:
                    fo.writelines(str(y))
            fo.write('..')
    """


    
    