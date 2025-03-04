import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

from torchvision import transforms
from mpl_toolkits.axes_grid1 import make_axes_locatable
import scipy.signal as signal


def my_readimage(image_path):
    '''
    读入图片:[0-255]array(256, 256, 3) ->[0,1]tensor torch.Size([1, 256, 256])
    '''
    imgcv = cv2.imread(image_path)
    transform = transforms.Compose([
    transforms.ToTensor()
])    
    imgcv = cv2.cvtColor(imgcv, cv2.COLOR_BGR2GRAY) #三通道转换为单通道
    imgcvb = transform(imgcv) #将一个取值范围在[0,255]的numpy.ndarray图像转换为[0,1.0]的torch.FloadTensor图像，同时各维度的顺序也将自动调整。

    return imgcvb

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


def my_save2image(matrix1, matrix2, image_path, cmap='viridis'):
    '''
    matrix1, matrix2: float32 [H,W] - 分别代表两个要显示的图像矩阵
    image_path: 保存图像的路径
    cmap: 颜色映射
    '''

    plt.clf()  # 清图。
    plt.cla()  # 清坐标轴
    plt.figure(figsize=(12, 6))  # 设定图像大小

    # 显示第一个图像
    plt.subplot(1, 2, 1)
    imgplot1 = plt.imshow(matrix1, cmap=cmap)
    plt.colorbar()  # 为第一个图像添加颜色条

    # 显示第二个图像
    plt.subplot(1, 2, 2)
    imgplot2 = plt.imshow(matrix2, cmap=cmap)
    plt.colorbar()  # 为第二个图像添加颜色条

    plt.savefig(image_path)  # 保存图像


def my_saveimage_plus(matrix,image_path):
          
    '''
    matrix:float32 [H,W]
    '''
    plt.figure(dpi=1000) # 清图。

    ax = plt.subplot()

    im = ax.imshow(matrix,resample=True)

    # create an Axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    plt.colorbar(im, cax=cax)
    plt.savefig(image_path)

      
def my_savetxt(matrix,txt_path):
    '''
    matrix:float32 [H,W]
    '''      

    np.savetxt(txt_path,matrix,fmt='%.10e',delimiter=",") #frame: 相位图 array:存入文件的数组


def my_readtxt(txt_path):
     matrix = np.loadtxt(txt_path,dtype=np.float32,delimiter=",") # frame:文件
     return matrix


def mkdir(path):
 
	folder = os.path.exists(path)
 
	if not folder:                   #判断是否存在文件夹如果不存在则创建为文件夹
		os.makedirs(path)            #makedirs 创建文件时如果路径不存在会创建这个路径
		print("---  new folder...  ---")
		print("---  OK  ---")
 
	else:
		print("---  There is this folder!  ---")

def visual_data(dataloader,root_path):

    for x,y in dataloader:
        print(f'shape of input [N,C,H,W]:{x.shape},{x.dtype} {x.max()}')
        print(f'shape of output:{y.shape},{x.dtype}')
        # print(f'all attribute of x:{dir(x)}')

        '''
        torchvision.utils.save_image(tensor, fp)
        # 参数
        # tensor(Tensor or list)：待保存的tensor数据（可以是上述处理好的grid）。如果给以一个四维的batch的tensor，将调用网格方法，然后再保存到本地。最后保存的图片是不带batch的。
        # fp：图片保存路径
        '''

        # torchvision.utils.save_image(x/x.max(),f'{root_path}/input.jpg')
        # torchvision.utils.save_image(y/y.max(),f'{root_path}/label.jpg')
        my_saveimage(x.reshape(x.shape[2],x.shape[3]),f'{root_path}/input.png')
        my_saveimage(y.reshape(x.shape[2],x.shape[3]),f'{root_path}/label.png')
        my_savetxt(x.reshape(x.shape[2],x.shape[3]),f'{root_path}/input.txt')
        my_savetxt(y.reshape(x.shape[2],x.shape[3]),f'{root_path}/label.txt')

        # print(f'label:{y}')

        break


def result_visual(pred_ref_path,pred_sam_path,gt_ref_path,gt_sam_path,save_path,cmap='viridis'):
    pred_ref = my_readtxt(pred_ref_path)
    gt_ref = my_readtxt(gt_ref_path)

    pred_sam = my_readtxt(pred_sam_path) 
    gt_sam = my_readtxt(gt_sam_path) 

    pred_sam_pred_ref = sam_ref(pred_sam,pred_ref)
    pred_sam_gt_ref = sam_ref(pred_sam,gt_ref)
    gt_sam_pred_ref = sam_ref(gt_sam,pred_ref)
    gt_sam_gt_ref = sam_ref(gt_sam,gt_ref)
    


    my_saveimage(pred_sam_pred_ref,f'{save_path}/pred_sam_pred_ref.png',cmap)
    my_saveimage(pred_sam_gt_ref,f'{save_path}/pred_sam_gt_ref.png',cmap)
    my_saveimage(gt_sam_pred_ref,f'{save_path}/gt_sam_pred_ref.png',cmap)
    my_saveimage(gt_sam_gt_ref,f'{save_path}/gt_sam_gt_ref.png',cmap)

    # my_saveimage(gt_sam_gt_ref,f'{save_path}/{cmap}_gt_sam_gt_ref.png',cmap)


def sam_ref(sam,ref,scale='pi'):
    """样品畸变相位减光纤畸变相位得到样品的相位

    Args:
        sam (numpy array): 样品畸变相位
        ref (numpy array): 光纤畸变相位
        scale (str, optional): 输入的是pi还是2pi. Defaults to 'pi'.

    Returns:
        numpy array: 样品的相位
    """  

    if scale=='pi':          
        diff = (sam - ref + 2.3)%(np.pi)
        diff = signal.medfilt(diff,(7,7)) #二维中值滤波   

    elif scale=='2pi':
        diff = (sam - ref + 4.1)%(np.pi*2)
        diff = signal.medfilt(diff,(11,11)) #二维中值滤波     

    return diff

def PaddingImage(img,original_width,original_height,target_width, target_height):
    """Pad the image with zeros to expand to a fixed size.

    Args:
        img (_type_): input little image
        original_width (_type_): width of original image
        original_height (_type_): height of original image
        target_width (_type_): width of target image
        target_height (_type_): height of target image

    Returns:
        extended_image: Image after pixel padding
    """    

    x_padding = target_width - original_width
    y_padding = target_height - original_height
    # 使用copyMakeBorder函数在原始图片的周围添加像素
    # blk_constant参数指定添加的像素颜色，这里是0（黑色）
    extended_image = cv2.copyMakeBorder(img, y_padding//2, y_padding//2, x_padding//2, x_padding//2, cv2.BORDER_CONSTANT, value=0)
    return extended_image   

def sam_ref_2pi(sam,ref):
    diff = np.mod(sam - ref,np.pi*2)
    return diff

def sam_ref_2pi_plot(sam,ref,save_path):
    diff = sam_ref_2pi(sam,ref)
    my_saveimage(diff,f'{save_path}/diff.png',dpi=800)
    my_savetxt(diff,f'{save_path}/diff.txt')
       



if __name__=='__main__':

    # cmaps = ['viridis', 'plasma', 'inferno', 'magma', 'cividis','Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
    #         'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
    #         'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn','binary', 'gist_yarg', 'gist_gray', 'gray', 'bone',
    #         'pink', 'spring', 'summer', 'autumn', 'winter', 'cool',
    #         'Wistia', 'hot', 'afmhot', 'gist_heat', 'copper','PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu', 'RdYlBu',
    #                   'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic','twilight', 'twilight_shifted', 'hsv','flag', 'prism', 'ocean', 'gist_earth', 'terrain',
    #                   'gist_stern', 'gnuplot', 'gnuplot2', 'CMRmap',
    #                   'cubehelix', 'brg', 'gist_rainbow', 'rainbow', 'jet',
    #                   'turbo', 'nipy_spectral', 'gist_ncar']

    ref_path = '/mnt/data/optimal/tangyuhang/workspace/iopen/ai4optical/phynet_git/result/1536_1536_ref_pi_01_prop_pi/2024-02-01-18-15/img_txt_folder/9000_pred.txt'
    sam_path=  '/mnt/data/optimal/tangyuhang/workspace/iopen/ai4optical/phynet_git/result/1536_1536_sam_pi_01_prop_pi/2024-02-01-18-53/img_txt_folder/4500_pred.txt'
    gt_ref_path = '/mnt/data/optimal/tangyuhang/workspace/iopen/ai4optical/phynet_git/traindata/gt/1536_1536_ref_pi_01_prop_pi.txt'
    gt_sam_path = '/mnt/data/optimal/tangyuhang/workspace/iopen/ai4optical/phynet_git/traindata/gt/1536_1536_sam_pi_01_prop_pi.txt'
    
    # ref = my_readtxt(sam_path)
    # sam = my_readtxt(gt_sam_path)   
    # diff = (sam - ref)
    save_path = '/mnt/data/optimal/tangyuhang/workspace/iopen/ai4optical/phynet_git/result_visual/baseline'
    # my_saveimage(diff,save_path)  

    # for cmap in tqdm(cmaps):
    #     result_visual(ref_path,sam_path,gt_ref_path,gt_sam_path,save_path,cmap)
    result_visual(ref_path,sam_path,gt_ref_path,gt_sam_path,save_path)
    # ref = my_readtxt(ref_path)
    # sam = my_readtxt(gt_sam_path)
    # diff = (sam - ref + 2.3)%(np.pi)

    # import scipy.signal as signal
    # diff = signal.medfilt(diff,(7,7)) #二维中值滤波
    # save_path = './ref_ref.png'
    # my_saveimage(diff,save_path)
    # dif = my_readtxt('/mnt/data/optimal/tangyuhang/workspace/iopen/ai4optical/phynet_git/traindata/gt/512_512_imagenet_prop_pi.txt')
    # dif640 = np.pad(dif,(64,64),'constant')

    # noise = (np.random.random((512,512))+np.random.random((512,512))+np.random.random((512,512)))/3
    # ref = noise*np.pi
    # ref640 = np.pad(ref,(64,64),'constant')

    # noise = my_readtxt('/mnt/data/optimal/tangyuhang/workspace/iopen/ai4optical/phynet_git/traindata/gt/1536_1536_phase_ref_prop_pi.txt')
    # ref = noise[768-256:768+256,768-256:768+256]
    # ref640 = np.pad(ref,(64,64),'constant')

    # sam640 = (dif640+ref640)/np.pi
    # # # print(ref.shape())
    # # # print(ref640.shape())
    # # my_saveimage(ref640,'./temp.png')
    # my_saveimage(dif640,'/mnt/data/optimal/tangyuhang/workspace/iopen/ai4optical/phynet_git/traindata/gt/dif640.png')
    # my_savetxt(dif640,'/mnt/data/optimal/tangyuhang/workspace/iopen/ai4optical/phynet_git/traindata/gt/dif640.txt')

    # my_saveimage(ref640,'/mnt/data/optimal/tangyuhang/workspace/iopen/ai4optical/phynet_git/traindata/gt/ref640.png')
    # my_savetxt(ref640,'/mnt/data/optimal/tangyuhang/workspace/iopen/ai4optical/phynet_git/traindata/gt/ref640.txt')
    
    # my_saveimage(sam640,'/mnt/data/optimal/tangyuhang/workspace/iopen/ai4optical/phynet_git/traindata/gt/sam640.png')
    # my_savetxt(sam640,'/mnt/data/optimal/tangyuhang/workspace/iopen/ai4optical/phynet_git/traindata/gt/sam640.txt')

