import numpy as np
import matplotlib.pyplot as plt
def my_savetxt(matrix,txt_path):
    '''
    matrix:float32 [H,W]
    '''      

    np.savetxt(txt_path,matrix,fmt='%.10e',delimiter=",") #frame: 相位图 array:存入文件的数组


def my_readtxt(txt_path):
     matrix = np.loadtxt(txt_path,dtype=np.float32,delimiter=",") # frame:文件
     return matrix


def my_saveimage(matrix,image_path,cmap='viridis',dpi=200):
          
    '''
    matrix:float32 [H,W]
    '''

    plt.clf() # 清图。
    plt.cla() # 清坐标轴

    plt.figure(figsize=(6, 6))
    imgplot = plt.imshow(matrix,cmap=cmap)
    plt.colorbar()
    plt.savefig(image_path,dpi=dpi)

def sam_ref_2pi(sam,ref):
    diff = np.mod(sam - ref,np.pi*2)
    return diff

def sam_ref_2pi_plot(sam,ref,save_path):
    diff = sam_ref_2pi(sam,ref)
    my_saveimage(diff,f'{save_path}/diff.png',dpi=800)
    my_savetxt(diff,f'{save_path}/diff.txt')



if __name__ == "__main__":
    ref = my_readtxt("/ailab/user/tangyuhang/LenslessFiberEndomicroscopicPhaseImaging/Result/ASNet/Sultimulate_U_Net/Simulate/3/2025-01-21-20-44/sam/img_txt_folder/2999_PredPhawrap2pi.txt")
    sam = my_readtxt("/ailab/user/tangyuhang/LenslessFiberEndomicroscopicPhaseImaging/Result/ASNet/Sultimulate_U_Net/Simulate/3/2025-01-21-20-44/ref/img_txt_folder/2999_PredPhawrap2pi.txt")
    savepath = "/ailab/user/tangyuhang/LenslessFiberEndomicroscopicPhaseImaging/ASNet" 

    sam_ref_2pi_plot(sam,ref-3.5,savepath)