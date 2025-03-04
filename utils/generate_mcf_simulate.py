import numpy as np
import matplotlib.pyplot as plt
import random
import os

import sys 
sys.path.append("/mnt/data/optimal/tangyuhang/workspace/iopen/ai4optical/phynet/option/simulate.yaml")


from os.path import join, getsize

import numpy as np
from library import my_saveimage,mkdir,my_savetxt
from prop import propcomplex
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import cv2
from config.parameter import Parameter,import_class
import argparse
from copy import deepcopy
import time
from library import (my_readtxt,mkdir,visual_data,my_saveimage,my_savetxt,my_save2image)
from dataset import mydataset
import torchvision
from torch.utils.tensorboard import SummaryWriter
from source_target_transforms import *
from utils.compute_metric import compute_core_std_plot
from torch.cuda.amp import autocast as autocast
def create_circles_in_rectangle_within_circle(height, width, a, b, number_of_cores, core_radius, fiber_radius,ispha='pha',scale = 'pi'):
    # 创建一个空白图像
    
    image = np.zeros((height, width))
    
    # 大圆的圆心和大圆内部像素值的设置
    fiber_center = (height // 2, width // 2)
    yy, xx = np.ogrid[:height, :width]
    distance_from_center = np.sqrt((xx - fiber_center[1])**2 + (yy - fiber_center[0])**2)
    is_inside_fiber = distance_from_center <= fiber_radius
    
    # 大圆内的像素赋予0到π的随机值
    if ispha=='amp':
        image[is_inside_fiber] = 0
    elif ispha=='pha':
        if scale == 'pi':
            image[is_inside_fiber] = np.random.uniform(0, np.pi, is_inside_fiber.sum())
        elif scale == '2pi':
            image[is_inside_fiber] = np.random.uniform(0, 2*np.pi, is_inside_fiber.sum())
        elif scale == 'semi_pi':
            image[is_inside_fiber] = np.random.uniform(0, np.pi/2, is_inside_fiber.sum())
    
    
    # 计算长方形内小圆的均匀分布
    rectangle_center = (height // 2, width // 2)
    rectangle_top_left = (rectangle_center[0] - a // 2, rectangle_center[1] - b // 2)
    
    # 长方形内小圆的间距
    cores_per_row = int(np.sqrt(number_of_cores * b / a))
    cores_per_col = number_of_cores // cores_per_row
    spacing_x = b // (cores_per_row + 1)
    spacing_y = a // (cores_per_col + 1)

    # 生成小圆，并赋予每个小圆内的像素相同的随机值
    for i in range(1, cores_per_col + 1):
        for j in range(1, cores_per_row + 1):
            center_x = rectangle_top_left[1] + j * spacing_x
            center_y = rectangle_top_left[0] + i * spacing_y
            # 判断小圆整体是否在大圆内
            if (np.abs(center_x - (fiber_center[1]))+9)**2 + (np.abs(center_y - (fiber_center[0]))+9)**2 <= fiber_radius**2:
                if ispha=='amp':
                    core_phase_value = 1
                elif ispha=='pha':
                    if scale == 'pi':
                        core_phase_value = np.random.uniform(0, np.pi)
                    elif scale == '2pi':
                        core_phase_value = np.random.uniform(0, 2*np.pi)
                    elif scale == 'semi_pi':
                        core_phase_value = np.random.uniform(0, np.pi/2)
                for y in range(-core_radius, core_radius + 1):
                    for x in range(-core_radius, core_radius + 1):
                        if x**2 + y**2 <= core_radius**2:
                            image[center_y + y, center_x + x] = core_phase_value
    
    return image

def mkdir(folder_path):
    if not os.path.isdir(folder_path):
        os.makedirs(folder_path)
        print("make dirs")

    else:
        print("dirs exists")

def getData(para):
    
    # 新参数
    num = para.num
    print(f'num of core:{num}')
    scale = para.scale
    print(f'scale of angle:{scale}')
    
    
    data = {
        1600: (1600, 800), #纤芯数量与光纤束的直径
        3000: (3000, 1096),
        6000: (6000, 1550),
        10000: (10000, 1400),
        15000: (15000, 2550),
        10:(10,100),
        100:(100,200),
        200:(200,282),
        500:(500,448),
        1000:(1000,650)
    }
    
    
    
    
    # 使用新参数
    rootpath = f'../simulateData/simulate_data/{para.constraint}/{para.scale}/{para.num}/{para.dist}/{para.fi}'
    
    # 相机的像素
    height = para.image_height#int(data[num][1]*para.fi)#para.image_height
    width = para.image_width#int(data[num][1]*para.fi)#para.image_width

    # 光纤的圆心和半径，决定了光纤间的间隙
    fiber_center = (width/2, height/2)
    fiber_radius = data[num][1]/2
    
    if data[num][0] < 100:
        
        fang = fiber_radius
        
    elif data[num][0] < 3000:
        fang = fiber_radius+10
        
    else:
        fang = fiber_radius+30

    a, b = int(fang*2), int(fang*2)  # 长方形的尺寸
    core_radius = 6
    number_of_cores = num
    
    mkdir(f"{rootpath}")
    
    pha = create_circles_in_rectangle_within_circle(height, width, a, b, number_of_cores, core_radius, fiber_radius,ispha='pha',scale=scale)
    # if para.before_resize['flag'] == True:
    #     pha = cv2.resize(pha, (para.resize['size'],para.resize['size']), interpolation=cv2.INTER_CUBIC)
    np.savetxt(f'{rootpath}/{number_of_cores}_pha_simulate.txt',pha,fmt='%.10e',delimiter=',')
    # 显示图像
    plt.figure(figsize=(6, 6))
    plt.imshow(pha, cmap='viridis')  # 使用HSV色彩映射以更好地显示相位
    plt.colorbar()
    plt.savefig(f'{rootpath}/{number_of_cores}_pha_simulate.png',dpi=800)#
    print('pha Done!!')

    amp = create_circles_in_rectangle_within_circle(height, width, a, b, number_of_cores, core_radius, fiber_radius,ispha='amp')
    # if para.before_resize['flag'] == True:
    #     amp = cv2.resize(amp, (para.resize['size'],para.resize['size']), interpolation=cv2.INTER_CUBIC)
    np.savetxt(f'{rootpath}/{number_of_cores}_amp_simulate.txt',amp,fmt='%.10e',delimiter=',')
    # 显示图像
    plt.figure(figsize=(6, 6))
    plt.imshow(amp, cmap='viridis')  # 使用HSV色彩映射以更好地显示相位
    plt.colorbar()
    plt.savefig(f'{rootpath}/{number_of_cores}_amp_simulate.png',dpi=800)#
    print('amp Done!!')

    # 2.光纤掩膜
    mask = deepcopy(pha)
    mask[mask < 0.02] = 0
    mask[mask > 0.0001] = 1
    kernel = np.ones((11,11),np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    my_saveimage(mask, f'{rootpath}/{number_of_cores}_mask_simulate.png')

    
    # generate speckle
    dist =  para.dist
    print(f'dist of prop:{dist}')
    pha = torch.tensor(pha)
    amp = torch.tensor(amp)
    
    Uo = amp*torch.exp(1j*pha) #光纤端面初始复光场
    Ui = propcomplex(Uo,dist=dist,device='cpu')

    speckle = torch.abs(Ui)

    dist_prop = str(dist).replace('.','')
    
    my_saveimage(speckle,f'{rootpath}/{number_of_cores}_speckle_prop{dist_prop}_simulate.png',dpi=800)
    my_savetxt(speckle,f'{rootpath}/{number_of_cores}_speckle_prop{dist_prop}_simulate.txt')
    print('speckle Done!!')  
    
    return pha.numpy(),amp.numpy(),mask,speckle.numpy()  

def get_speckle(para,pha,amp):
    # generate speckle
    dist =  para.dist
    print(f'dist of prop:{dist}')
    pha = torch.tensor(pha)
    amp = torch.tensor(amp)
    print(torch.max(pha))
    
    Uo = amp*torch.exp(1j*pha) #光纤端面初始复光场
    Ui = propcomplex(Uo,dist=dist,device='cpu')

    speckle = torch.abs(Ui)
    
    print('speckle Done!!')

    return speckle.numpy()

def get_speckle2(dist,pha,amp):
    # generate speckle

    # print(f'2ed dist of prop:{dist}')
    pha = torch.tensor(pha)
    amp = torch.tensor(amp)
    print(torch.max(pha))
    
    Uo = amp*torch.exp(1j*pha) #光纤端面初始复光场
    Ui = propcomplex(Uo,dist=dist,device='cpu')

    speckle = torch.abs(Ui)
    
    print("-"*10+f'speckle at {dist} Done!!')

    return speckle.numpy()

def mcf_simulate(para,data=None):
    
    # 新参数
    num = para.num
    print(f'num of core:{num}')
    scale = para.scale
    print(f'scale of angle:{scale}')
    
    if data is None:    
        data = {
            1600: (1600, 800), #纤芯数量与光纤束的直径
            # 3000: (3000, 1096),#纤芯半径是9
            3000: (3000, 766),#纤芯半径是6
            6000: (6000, 1550),
            10000: (10000, 1400),
            # 15000: (15000, 2550), #纤芯半径是9
            15000: (15000, 1708),
            10:(10,100),
            100:(100,200),
            200:(200,282),
            500:(500,448),
            1000:(1000,650)
        }
        
    else:
        data = data
        
    # 相机的像素
    
    if para.isfi == True:
        
        height = int(data[num][1]*para.fi)#para.image_height
        width = int(data[num][1]*para.fi)#para.image_width
        print(f'图片像素是光纤束的{para.fi}倍')
    else:        
        height = para.image_height#int(data[num][1]*para.fi)#para.image_height
        width = para.image_width#int(data[num][1]*para.fi)#para.image_width
        print(f'图片像素是光纤束的{height}_{width}倍')

    # 光纤的圆心和半径，决定了光纤间的间隙
    fiber_center = (width/2, height/2)
    fiber_radius = data[num][1]/2
    
    if data[num][0] < 100:
        
        fang = fiber_radius
        
    elif data[num][0] < 3000:
        fang = fiber_radius+10
        
    else:
        fang = fiber_radius+30

    a, b = int(fang*2), int(fang*2)  # 长方形的尺寸
    core_radius = para.core_radius
    number_of_cores = num
    

    
    pha = create_circles_in_rectangle_within_circle(height, width, a, b, number_of_cores, core_radius, fiber_radius,ispha='pha',scale=scale)
    if para.before_resize['flag'] == True:
        pha = cv2.resize(pha, (para.before_resize['size'],para.before_resize['size']), interpolation=cv2.INTER_CUBIC)
    print('pha Done!!')

    amp = create_circles_in_rectangle_within_circle(height, width, a, b, number_of_cores, core_radius, fiber_radius,ispha='amp')
    if para.before_resize['flag'] == True:
        amp = cv2.resize(amp, (para.before_resize['size'],para.before_resize['size']), interpolation=cv2.INTER_CUBIC)
    print('amp Done!!')

    # 2.光纤掩膜
    mask = deepcopy(pha)
    mask[mask < 0.02] = 0
    mask[mask > 0.0001] = 1
    kernel = np.ones((11,11),np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


    
    # generate speckle
    dist =  para.dist
    print(f'dist of prop:{dist}')
    pha = torch.tensor(pha)
    amp = torch.tensor(amp)
    
    Uo = amp*torch.exp(1j*pha) #光纤端面初始复光场
    Ui = propcomplex(Uo,dist=dist,device='cpu')

    speckle = torch.abs(Ui)
    
    print('speckle Done!!')  
    
    return pha.numpy(),amp.numpy(),mask,speckle.numpy()  

def mcf_simulate_v2(para,data=None):
    '''
    input:
        para.isfi:图像尺寸是光纤束尺寸的几倍，如果为False则是固定值。
        para.image_height：图像的高，在第一个参数
        para.image_width
        para.core_radius：每个纤芯的半径

    output：
        pha：生成光纤自身相位畸变，numpy
        amp：光纤自生振幅图像，
        mask：光纤端面mask
    '''
    
    # 新参数
    num = para.num
    print(f'num of core:{num}')
    scale = para.scale
    print(f'scale of angle:{scale}')
    
    if data is None:
            #光纤束半径计算，纤芯数量开根号，乘（纤芯半径+1）乘2 
        data = {
            1600: (1600, 800), #纤芯数量与光纤束的直径
            # 3000: (3000, 1096),#纤芯半径是9
            3000: (3000, 766),#纤芯半径是6
            6000: (6000, 1550),
            10000: (10000, 1400),
            # 15000: (15000, 2550), #纤芯半径是9
            # 15000: (15000, 1708),#六个距离加1
            # 15000: (15000, 980), #纤芯半径是4加上0间距
            # 15000: (15000, 1250), #纤芯半径是4加上一个间距
            15000: (15000, 1470), #纤芯半径是5加上一个间距或者6加0个间距
            # 15000: (15000, 1400), #纤芯半径是5加上0.8间距
            # 15000: (15000, 1250), #纤芯半径是5加上0间距
            10:(10,100),
            100:(100,200),
            200:(200,282),
            500:(500,448),
            1000:(1000,650)
        }
        
    else:
        data = data
        
    # 相机的像素
    
   
    height = para.image_height#int(data[num][1]*para.fi)#para.image_height
    width = para.image_width#int(data[num][1]*para.fi)#para.image_width
    print(f'图片像素是光纤束的{height}_{width}倍')

    # 光纤的圆心和半径，决定了光纤间的间隙
    fiber_center = (width/2, height/2)
    fiber_radius = data[num][1]/2
    
    if data[num][0] < 100:
        
        fang = fiber_radius
        
    elif data[num][0] < 3000:
        fang = fiber_radius+10
        
    else:
        fang = fiber_radius+30

    a, b = int(fang*2), int(fang*2)  # 长方形的尺寸
    core_radius = para.core_radius
    number_of_cores = num
    

    
    pha = create_circles_in_rectangle_within_circle(height, width, a, b, number_of_cores, core_radius, fiber_radius,ispha='pha',scale=scale)

    print('pha Done!!')

    amp = create_circles_in_rectangle_within_circle(height, width, a, b, number_of_cores, core_radius, fiber_radius,ispha='amp')

    print('amp Done!!')

    # 2.光纤掩膜
    mask = deepcopy(pha)
    mask[mask < 0.02] = 0
    mask[mask > 0.0001] = 1
    kernel = np.ones((11,11),np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
   
    print('speckle Done!!')  
    
    return pha,amp,mask

def mcf_simulate_plot_v2(para,data=None):
    
    pha,amp,mask = mcf_simulate_v2(para,data=None)

    # 使用新参数
    rootpath = f'/ailab/user/tangyuhang/ws/Traindata/simulate_data/{para.scale}/{para.num}/{para.core_radius}/{para.image_height}-{para.image_width}'
    
    mkdir(f"{rootpath}")

    np.savetxt(f'{rootpath}/{para.num}_pha_simulate.txt',pha,fmt='%.10e',delimiter=',')
    # 显示图像
    plt.figure(figsize=(6, 6))
    plt.imshow(pha, cmap='viridis')  # 使用HSV色彩映射以更好地显示相位
    plt.colorbar()
    plt.savefig(f'{rootpath}/{para.num}_pha_simulate.png',dpi=800)#
    print('pha Done!!')       
 
    np.savetxt(f'{rootpath}/{para.num}_amp_simulate.txt',amp,fmt='%.10e',delimiter=',')
    # 显示图像
    plt.figure(figsize=(6, 6))
    plt.imshow(amp, cmap='viridis')  # 使用HSV色彩映射以更好地显示相位
    plt.colorbar()
    plt.savefig(f'{rootpath}/{para.num}_amp_simulate.png',dpi=800)#
    print('amp Done!!')
    
    np.savetxt(f'{rootpath}/{para.num}_mask_simulate.txt',mask,fmt='%.10e',delimiter=',')
    my_saveimage(mask, f'{rootpath}/{para.num}_mask_simulate.png')
    print('mask Done!!')  
    
    return pha,amp,mask 

def mcf_simulate_plot(para,data=None):
    
    pha,amp,mask,speckle = mcf_simulate(para,data=None)

    # 使用新参数
    rootpath = f'/ailab/user/tangyuhang/ws/Traindata/simulate_data/{para.constraint}/{para.scale}/{para.num}/{para.dist}/{para.image_width}'
    
    mkdir(f"{rootpath}")

    np.savetxt(f'{rootpath}/{para.num}_pha_simulate.txt',pha,fmt='%.10e',delimiter=',')
    # 显示图像
    plt.figure(figsize=(6, 6))
    plt.imshow(pha, cmap='viridis')  # 使用HSV色彩映射以更好地显示相位
    plt.colorbar()
    plt.savefig(f'{rootpath}/{para.num}_pha_simulate.png',dpi=800)#
    print('pha Done!!')       
 
    np.savetxt(f'{rootpath}/{para.num}_amp_simulate.txt',amp,fmt='%.10e',delimiter=',')
    # 显示图像
    plt.figure(figsize=(6, 6))
    plt.imshow(amp, cmap='viridis')  # 使用HSV色彩映射以更好地显示相位
    plt.colorbar()
    plt.savefig(f'{rootpath}/{para.num}_amp_simulate.png',dpi=800)#
    print('amp Done!!')
    
    my_saveimage(mask, f'{rootpath}/{para.num}_mask_simulate.png')
    print('amp Done!!')
    
    dist_prop = str(para.dist).replace('.','')
    
    my_saveimage(speckle,f'{rootpath}/{para.num}_speckle_prop{dist_prop}_simulate.png',dpi=800)
    my_savetxt(speckle,f'{rootpath}/{para.num}_speckle_prop{dist_prop}_simulate.txt')
    print('speckle Done!!')    
    
    return pha,amp,mask,speckle   
    
def mcf_simulate_2dist_plot(para,data=None):
    
    pha,amp,mask,speckle = mcf_simulate(para,data=None)
    speckle_gt2 = get_speckle2(para.dist2,pha,amp)

    # 使用新参数
    rootpath = f'/ailab/user/tangyuhang/ws/Traindata/simulate_data/{para.constraint}/{para.scale}/{para.num}/{para.dist}/{para.image_width}'
    
    mkdir(f"{rootpath}")

    np.savetxt(f'{rootpath}/{para.num}_pha_simulate.txt',pha,fmt='%.10e',delimiter=',')
    # 显示图像
    plt.figure(figsize=(6, 6))
    plt.imshow(pha, cmap='viridis')  # 使用HSV色彩映射以更好地显示相位
    plt.colorbar()
    plt.savefig(f'{rootpath}/{para.num}_pha_simulate.png',dpi=800)#
    print('pha Done!!')       
 
    np.savetxt(f'{rootpath}/{para.num}_amp_simulate.txt',amp,fmt='%.10e',delimiter=',')
    # 显示图像
    plt.figure(figsize=(6, 6))
    plt.imshow(amp, cmap='viridis')  # 使用HSV色彩映射以更好地显示相位
    plt.colorbar()
    plt.savefig(f'{rootpath}/{para.num}_amp_simulate.png',dpi=800)#
    print('amp Done!!')
    
    my_saveimage(mask, f'{rootpath}/{para.num}_mask_simulate.png')
    print('amp Done!!')
    
    dist_prop = str(para.dist).replace('.','')
    
    my_saveimage(speckle,f'{rootpath}/{para.num}_speckle_prop{dist_prop}_simulate.png',dpi=800)
    my_savetxt(speckle,f'{rootpath}/{para.num}_speckle_prop{dist_prop}_simulate.txt')
    print('speckle Done!!')    
    
    dist2_prop = str(para.dist2).replace('.','')
    
    my_saveimage(speckle_gt2,f'{rootpath}/{para.num}_speckle_prop{dist2_prop}_simulate.png',dpi=800)
    my_savetxt(speckle_gt2,f'{rootpath}/{para.num}_speckle_prop{dist2_prop}_simulate.txt')
    print('speckle2 Done!!')        

    return pha,amp,mask,speckle,speckle_gt2       

import matplotlib.pyplot as plt
import matplotlib
from matplotlib import animation   
from celluloid import Camera
if __name__ == '__main__':



    ims = []
    
    parser = argparse.ArgumentParser(description='Training script')
    parser.add_argument('--opt', type=str, default='D:\\tyh\phynet\option\simulate.yaml', help='Path to the configuration file')
    args = parser.parse_args()
    para = Parameter(args.opt)
    
    loops = [0.6,0.5,0.4,0.3,0.2,0.1,0.09,0.08,0.07,0.06,0.05,0.04,0.03,0.02,0.01,0.005,0.001]  
    plt.ion() #开启interactive mode 
    # plt.tight_layout()
    fig = plt.figure(figsize=(8, 8))
    
    
    for dist in loops:
        para.dist = dist
        
    
        pha_gt,amp_gt,mask_gt,speckle_gt  = mcf_simulate(para)
        

        # plt.figure(figsize=(12, 12))
        plt.clf()
        plt.subplot(2, 2, 1)
        plt.imshow(pha_gt, cmap='viridis')  # 使用HSV色彩映射以更好地显示相位
        
        plt.colorbar()
        plt.title('pha_gt')
        
        plt.subplot(2, 2, 2)
        plt.imshow(amp_gt, cmap='viridis')  # 使用HSV色彩映射以更好地显示相位
        plt.colorbar()
        plt.title('amp_gt')
        
        plt.subplot(2, 2, 3)
        plt.imshow(mask_gt, cmap='viridis')  # 使用HSV色彩映射以更好地显示相位
        plt.colorbar()
        plt.title('mask_gt')
        
        plt.subplot(2, 2, 4)
        plt.imshow(speckle_gt, cmap='viridis')  # 使用HSV色彩映射以更好地显示相位
        plt.colorbar()
        plt.title(f'speckle_gt:{dist}')
        
        plt.savefig(f'.\\10000_{dist}_mcf_simulate.png')  
        plt.pause(1)
        

          
    plt.ioff()      #关闭interactive mode
    # plt.show()

        