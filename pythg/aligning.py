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
    plt.title("imging of fluo, %d"%(np.max(fluo)))
    plt.axis('off')
    plt.show()
    print(fluo.shape)
    return fluo

def points_distance(a,b):
    x1,y1 = a[0], a[1]
    x2,y2 = b[0], b[1]
    return np.sqrt((x1-x2)**2 + (y1-y2)**2)

def read_contours(file_path):
    with open(file_path,'r') as fo:
        date = fo.read()
        date = date.replace("["," ")
        date = date.replace("]"," ")
        date = date.split('..')
        data = [np.array(i.split(),dtype=np.int32).reshape(-1,1,2) for i in date]
    return data

def save_aliging(file1,file2,new_path):
    distance_matrix = np.zeros((file1.shape[0],file2.shape[0]))
    for i in range(distance_matrix.shape[0]):
        for j in range(distance_matrix.shape[1]):
            distance_matrix[i,j] = points_distance(file1[i,:2],file2[j,:2])
           
    index_list = []
    for i in np.arange(distance_matrix.shape[0]):
        min_j = np.argmin(distance_matrix[i,:])
        if distance_matrix[i,min_j] < np.min([file1[i,2],file2[min_j,2]]):
            index_list.append([i,min_j])
    
    col_list = []
    for i,j in index_list:
        col = np.concatenate((file1[i].reshape(1,-1),file2[j].reshape(1,-1)),axis=1)
        col_list.append(col)
        
    file_merge = np.concatenate(col_list,axis=0)
    
    head = "x,y,radius,intensities,area,arc,x1,y1,radius1,intensities1,area1,arc1"
    np.savetxt(new_path,file_merge, fmt='%.4f',delimiter=',',header=head)

def draw_contours(img1,img2,contours1,contours2,fluo):
    new_image2 = np.zeros((fluo.shape[0],fluo.shape[1],3),dtype='int')
    
    for i,cnt in enumerate(contours1):
        cv2.drawContours(new_image2, contours1, i, (255,0,0), 1)
        
    for i,cnt in enumerate(contours1):
        cv2.drawContours(new_image2, contours2, i, (0,255,0), 1)
        
    plt.figure()
    plt.imshow(new_image2)
    plt.title("red is %s, green is %s"%(img1,img2))