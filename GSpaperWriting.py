import numpy as np
import matplotlib.pyplot as plt

import sys
code_path =  '/ailab/user/tangyuhang/LenslessFiberEndomicroscopicPhaseImaging/ASNet'
sys.path.append(code_path)

from library import my_saveimage,mkdir,my_savetxt,sam_ref_2pi_plot
from tqdm import tqdm
import cv2
from config.parameter import Parameter
import argparse
from copy import deepcopy
import time
from utils.compute_metric import compute_core_std_plot
from torch.cuda.amp import autocast as autocast
from utils.generate_mcf_simulate import mcf_simulate_plot_v2,get_speckle2,get_speckle
from TrainUnetSultimulate1Dist import get_RefSamData,plotresult
from TrainExpHela4Dist import get_RefSamRealData
def prop(H, dx, dy, lambd, dist):
    Ny, Nx = H.shape
    fft_H = np.fft.fftshift(np.fft.fft2(H))

    x, y = np.meshgrid(np.arange(1 - Nx / 2, Nx / 2 + 1), np.arange(1 - Ny / 2, Ny / 2 + 1))
    r = (2 * np.pi * x / (dx * Nx)) ** 2 + (2 * np.pi * y / (dy * Ny)) ** 2

    k = 2 * np.pi / lambd
    kernel = np.exp(-1j * np.sqrt(k ** 2 - r) * dist)

    fft_HH = fft_H * kernel
    fft_HH = np.fft.ifftshift(fft_HH)

    U = np.fft.ifft2(fft_HH)
    P = np.angle(U)

    return U, P

def prop_v2(H, dist, dx=2e-6, dy=2e-6, lambd=532e-9):
    Ny, Nx = H.shape
    fft_H = np.fft.fftshift(np.fft.fft2(H))

    x, y = np.meshgrid(np.arange(1 - Nx / 2, Nx / 2 + 1), np.arange(1 - Ny / 2, Ny / 2 + 1))
    r = (2 * np.pi * x / (dx * Nx)) ** 2 + (2 * np.pi * y / (dy * Ny)) ** 2

    k = 2 * np.pi / lambd
    kernel = np.exp(-1j * np.sqrt(k ** 2 - r) * dist)

    fft_HH = fft_H * kernel
    fft_HH = np.fft.ifftshift(fft_HH)

    U = np.fft.ifft2(fft_HH)
    P = np.angle(U)

    return U, P

def gsplot(img,amp,speckle,para):

    localtime = time.strftime("%Y-%m-%d-%H-%M", time.localtime())    
    result_folder = f'/ailab/user/tangyuhang/ws/Comparative_methods/GSmethod/result/{para.scale}/{para.dist}/{localtime}'
    img_txt_folder = f'{result_folder}/img_txt_folder' 
    mkdir(img_txt_folder)
    # Load image
    # img = np.loadtxt(para.datapath['pha'], delimiter=',')
    img_convert = img.astype(np.float64)
    img_phase = img_convert * 1

    # 2.光纤掩膜
    mask = deepcopy(img_phase)
    mask[mask < 0.02] = 0
    mask[mask > 0.0001] = 1
    kernel = np.ones((11,11),np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


    # Load amp
    # amp = np.loadtxt(para.datapath['amp_gt'], delimiter=',')
    amp_convert = amp.astype(np.float64)
    amp_phase = amp_convert * 1


    # Parameters
    dx = 2e-6
    dy = 2e-6
    lambd = 532e-9
    z = para.dist

    # Propagation (仿真散斑)
    # U = amp_phase * np.exp(1j * img_phase)
    # U_prop, _ = prop(U, dx, dy, lambd, z)

    # speckle = np.loadtxt(para.datapath['speckle'], delimiter=',')

    # Reconstruct the phase on the facet with sample from the diffraction pattern
    target_intensity = amp_phase
    rand_phi = np.zeros_like(speckle)
    source_intensity = speckle
    source = source_intensity * np.exp(1j * rand_phi)
    phase_error = []

    for step in range(3000):
        print(f'Iteration {step + 1}')

        source = source_intensity * np.exp(1j * np.angle(source))

        target, _ = prop(source, dx, dy, lambd, -z)
        angle_target = np.angle(target)
        phase_diff = np.angle(np.exp(1j * (angle_target - img_phase)))

        target = target_intensity * np.exp(1j * angle_target)

        source, _ = prop(target, dx, dy, lambd, z)

        source_phase_full = np.angle(source)  # store the calculated phase
        phase_error.append(np.mean(np.abs(angle_target - img_phase)))

        plt.figure(2364)
        plt.subplot(2, 2, 1)
        plt.imshow(np.angle(np.exp(1j * angle_target)))
        plt.axis('off')
        plt.title('Phase Reconstruction')

        plt.subplot(2, 2, 2)
        plt.imshow(phase_diff)
        plt.axis('off')
        plt.title('Phase Difference')

        plt.subplot(2, 2, (3,4))
        plt.plot(phase_error)
        plt.title('Phase Error')

        # plt.pause(0.02)




    plt.savefig(f'{img_txt_folder}/result.png')

    dpi = 400
    my_saveimage(np.mod(angle_target,2*np.pi),f'{img_txt_folder}/{step}_PredPhawrap2pi.png',dpi=dpi)
    my_savetxt(np.mod(angle_target,2*np.pi),f'{img_txt_folder}/{step}_PredPhawrap2pi.txt')

    my_saveimage((angle_target),f'{img_txt_folder}/{step}_PredPha.png',dpi=dpi)
    my_savetxt((angle_target),f'{img_txt_folder}/{step}_PredPha.txt')

    my_saveimage(np.abs(source),f'{img_txt_folder}/{step}_PredAmp.png',dpi=dpi)
    my_savetxt(np.abs(source),f'{img_txt_folder}/{step}_PredAmp.txt')

    my_saveimage(np.abs(source)-speckle,f'{img_txt_folder}/{step}_AmpLoss.png',dpi=dpi)
    my_savetxt(np.abs(source)-speckle,f'{img_txt_folder}/{step}_AmpLoss.txt')
    
    my_saveimage(np.mod(angle_target - img_phase,2*np.pi),f'{img_txt_folder}/{step}_PhaLoss.png',dpi=dpi)
    my_savetxt(np.mod(angle_target - img_phase,2*np.pi),f'{img_txt_folder}/{step}_PhaLoss.txt')
    
    compute_core_std_plot(amp_phase,np.mod(angle_target - img_phase,2*np.pi),f'{img_txt_folder}/{step}core_std.png',meanflag=True,labeledflag=True)
    # compute_core_std_plot(amp_gt.cpu().detach().numpy(),np.mod(flattened_pred_pha.cpu().detach().numpy(),2*np.pi),f'{img_txt_folder}/{step}core_std.png',outputflag=True)
    my_saveimage(np.mod((amp_phase*angle_target),2*np.pi),f'{img_txt_folder}/{step}_Phamulmask.png',dpi=dpi)
    plt.close()    

def testGS():
    parser = argparse.ArgumentParser(description='Training script')
    parser.add_argument('--opt', type=str, default=f'{code_path}/option/GSpaperWriting.yaml', help='Path to the configuration file')
    args = parser.parse_args()
    para = Parameter(args.opt)

    img = cv2.imread('/ailab/user/tangyuhang/ws/Traindata/img/cityscapes_2.jpg')
    img = img[0:512,0:512]
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_img = gray_img.astype('float32')
    gray_img = gray_img/256.0
    gray_img = gray_img - 0.5
    if para.scale == 'pi':
        gray_img = gray_img *  np.pi 

    elif para.scale == '2pi':
        gray_img = gray_img *  np.pi * 2

    my_saveimage(gray_img,'/ailab/user/tangyuhang/ws/Traindata/img/cityscapes_2_pi.jpg')

    amp = np.ones((512, 512), dtype=float)
    my_saveimage(amp,'/ailab/user/tangyuhang/ws/Traindata/img/cityscapes_2_amp.jpg')

    speckle = get_speckle(para,gray_img,amp)
    my_saveimage(speckle,f'/ailab/user/tangyuhang/ws/Traindata/img/cityscapes_2_speckle.jpg')

    gsplot(gray_img,amp,speckle,para)

    print('done')

def get4speckle(para,pha,amp):

    ref_speckle1 = get_speckle2(para.dist1,pha,amp)
    ref_speckle2 = get_speckle2(para.dist2,pha,amp)
    ref_speckle3 = get_speckle2(para.dist3,pha,amp)
    ref_speckle4 = get_speckle2(para.dist4,pha,amp)    

    return ref_speckle1,ref_speckle2,ref_speckle3,ref_speckle4

def get4speckle_plot(para,pha,amp,save_path):

    # Generate different speckle at different distances
    ref_speckle1,ref_speckle2,ref_speckle3,ref_speckle4 = get4speckle(para,pha,amp)

    plt.figure()
    plt.imshow(ref_speckle1)
    plt.savefig(f'{save_path}/ref_speckle1.png')

    plt.figure()
    plt.imshow(ref_speckle2)
    plt.savefig(f'{save_path}/ref_speckle2.png')

    plt.figure()
    plt.imshow(ref_speckle3)
    plt.savefig(f'{save_path}/ref_speckle3.png')

    plt.figure()
    plt.imshow(ref_speckle4)
    plt.savefig(f'{save_path}/ref_speckle4.png')

    np.savetxt(f'{save_path}/ref_speckle1.txt',ref_speckle1,fmt='%.6e',delimiter=",")
    np.savetxt(f'{save_path}/ref_speckle2.txt',ref_speckle2,fmt='%.6e',delimiter=",")
    np.savetxt(f'{save_path}/ref_speckle3.txt',ref_speckle3,fmt='%.6e',delimiter=",")
    np.savetxt(f'{save_path}/ref_speckle4.txt',ref_speckle4,fmt='%.6e',delimiter=",")

    return ref_speckle1,ref_speckle2,ref_speckle3,ref_speckle4

def sourceToTargetInit(ref_amp_gt,ref_speckles):
    target_intensity = ref_amp_gt
    angle_target_list = []

    rand_phi = np.zeros_like(ref_amp_gt)
    source_intensitys = ref_speckles
    source = source_intensitys[0] * np.exp(1j * rand_phi)

    return target_intensity,angle_target_list,source

def sourceToTarget(source,source_intensity,target_intensity,z):
    # Parameters
    dx = 2e-6
    dy = 2e-6
    lambd = 532e-9
    source = source_intensity * np.exp(1j * np.angle(source))

    target, _ = prop(source, dx, dy, lambd, -z)
    angle_target = np.angle(target)
    target = target_intensity * np.exp(1j * angle_target)

    source, _ = prop(target, dx, dy, lambd, z)

    return source,target

def targetToSource(target,target_intensity,angle_target_list,zs,source_intensitys,img_txt_folder,epochs=300):
    print(f"target_intensity type: {type(target_intensity)}, dtype: {getattr(target_intensity, 'dtype', 'Not an array')}")
    phase_error = []

    # Record the start time
    start_time = time.time()

    for step in tqdm(range(epochs)):
        # print(f'Iteration {step + 1}')

        for i in (range(len(zs))):
            # print(i)

            source, _ = prop_v2(target, zs[i])
            source = source_intensitys[i] * np.exp(1j * np.angle(source))

            target, _ = prop_v2(source, -zs[i])
            angle_target = np.angle(target)

            target = target_intensity * np.exp(1j * angle_target)

            angle_target_list[i] = angle_target

        angle_target = np.mean(angle_target_list,0)
        target = target_intensity * np.exp(1j * angle_target)



        phase_diff = angle_target - ref_pha_gt
        phase_error.append(np.mean(np.abs(np.mod(phase_diff,2*np.pi))))
        plt.figure(2364)
        plt.subplot(2, 2, 1)
        plt.imshow(np.angle(np.exp(1j * angle_target)))
        plt.axis('off')
        plt.title('Phase Reconstruction')

        plt.subplot(2, 2, 2)
        plt.imshow(np.mod(phase_diff,2*np.pi))
        plt.axis('off')
        plt.title('Phase Difference')

        plt.subplot(2, 2, (3,4))
        plt.plot(phase_error)
        plt.title('Phase Error')

    # Record the end time
    end_time = time.time()

    # Calculate the time taken
    time_taken = end_time - start_time
    print(f"Time taken to run the program: {time_taken} seconds")
    print(f"Time taken to run the program in a loop: {time_taken/epochs} seconds")

    # Save the time taken to a file
    with open(f"{img_txt_folder}/time_log.txt", "w") as file:
        file.write(f"Time taken to run the program: {time_taken} seconds")
        file.write(f"epochs: {epochs}")
        file.write(f"Time taken to run the program in a loop: {time_taken/epochs} seconds")


    plt.savefig(f'{img_txt_folder}/result.png',dpi=1000)

    dpi = 400
    my_saveimage(np.mod(angle_target,2*np.pi),f'{img_txt_folder}/{step}_PredPhawrap2pi.png',dpi=dpi)
    my_savetxt(np.mod(angle_target,2*np.pi),f'{img_txt_folder}/{step}_PredPhawrap2pi.txt')

    my_saveimage((angle_target),f'{img_txt_folder}/{step}_PredPha.png',dpi=dpi)
    my_savetxt((angle_target),f'{img_txt_folder}/{step}_PredPha.txt')
    
    my_saveimage(np.mod(phase_diff,2*np.pi),f'{img_txt_folder}/{step}_PhaLoss.png',dpi=dpi)
    my_savetxt(np.mod(phase_diff,2*np.pi),f'{img_txt_folder}/{step}_PhaLoss.txt')
    
    compute_core_std_plot(target_intensity,np.mod(phase_diff,2*np.pi),f'{img_txt_folder}/{step}core_std.png',meanflag=True,labeledflag=True)
    my_saveimage(np.mod((target_intensity*angle_target),2*np.pi),f'{img_txt_folder}/{step}_Phamulmask.png',dpi=dpi)
    plt.close()    

    return angle_target


def get_speckles(para,pha_gt,amp_gt,sam_pha_gt,sam_amp_gt,result_folder):
    speckle_gt =  get_speckle2(para.dist,pha_gt,amp_gt)
    speckle_gt2 = get_speckle2(para.dist2,pha_gt,amp_gt)
    speckle_gt3 = get_speckle2(para.dist3,pha_gt,amp_gt)
    speckle_gt4 = get_speckle2(para.dist4,pha_gt,amp_gt)

    sam_speckle_gt =  get_speckle2(para.dist,sam_pha_gt,sam_amp_gt)
    sam_speckle_gt2 = get_speckle2(para.dist2,sam_pha_gt,sam_amp_gt)
    sam_speckle_gt3 = get_speckle2(para.dist3,sam_pha_gt,sam_amp_gt)
    sam_speckle_gt4 = get_speckle2(para.dist4,sam_pha_gt,sam_amp_gt)

    plotresult(pha_gt,f'{result_folder}/ref_pha_gt_mul_mask',dpi=1000)
    plotresult(amp_gt,f'{result_folder}/ref_amp_gt_mul_mask',dpi=1000)
    plotresult(speckle_gt,f'{result_folder}/ref_speckle_gt')
    plotresult(speckle_gt2,f'{result_folder}/ref_speckle_gt2')
    plotresult(speckle_gt3,f'{result_folder}/ref_speckle_gt3')
    plotresult(speckle_gt4,f'{result_folder}/ref_speckle_gt4')

  
    

    plotresult(sam_pha_gt,f'{result_folder}/sam_pha_gt_mul_mask',dpi=1000)
    plotresult(sam_amp_gt,f'{result_folder}/sam_amp_gt_mul_mask',dpi=1000)
    plotresult(sam_speckle_gt,f'{result_folder}/sam_speckle_gt')
    plotresult(sam_speckle_gt2,f'{result_folder}/sam_speckle_gt2')
    plotresult(sam_speckle_gt3,f'{result_folder}/sam_speckle_gt3')
    plotresult(sam_speckle_gt4,f'{result_folder}/sam_speckle_gt4')

    return speckle_gt,speckle_gt2,speckle_gt3,speckle_gt4,sam_speckle_gt,sam_speckle_gt2,sam_speckle_gt3,sam_speckle_gt4


if __name__=="__main__":

    parser = argparse.ArgumentParser(description='Training script')
    parser.add_argument('--opt', type=str, default=f'{code_path}/option/GSpaperWriting.yaml', help='Path to the configuration file')
    args = parser.parse_args()
    para = Parameter(args.opt)

    localtime = time.strftime("%Y-%m-%d-%H-%M", time.localtime())    
    result_folder = f'/ailab/user/tangyuhang/LenslessFiberEndomicroscopicPhaseImaging/Result/GSmethod/{para.RealOrSimulate}_{para.WhichData}/{para.scale}/{localtime}'
    img_txt_folder = f'{result_folder}/img_txt_folder' 
    mkdir(result_folder)
    mkdir(img_txt_folder)
    mkdir(f'{img_txt_folder}/ref')
    mkdir(f'{img_txt_folder}/sam')
    # Generate the phase distortion of fiber


    # generate data
    if para.RealOrSimulate == 'Simulate':
        ref_pha_gt,ref_amp_gt,mask,diff_pha_gt = get_RefSamData(para)
        ref_pha_gt = ref_pha_gt*mask
        ref_amp_gt = ref_amp_gt*mask     

        # load the phase of the sample distortion
        sam_pha_gt = np.mod(ref_pha_gt+diff_pha_gt,2*np.pi)    
        sam_amp_gt = ref_amp_gt

        sam_pha_gt = sam_pha_gt*mask
        sam_amp_gt = sam_amp_gt*mask   
        print(np.max(diff_pha_gt))
        plotresult(diff_pha_gt,f'{result_folder}/sample_pha')
    elif para.RealOrSimulate == 'Real':
        pha_gt,amp_gt,mask,sam_pha_gt,sam_amp_gt = get_RefSamRealData(para)

        ref_pha_gt = pha_gt
        ref_amp_gt = amp_gt
        sam_pha_gt = sam_pha_gt
        sam_amp_gt = sam_amp_gt


    else:
        raise ValueError("RealOrSimulate 必须是Real Or Simulate")    




    ref_speckle1,ref_speckle2,ref_speckle3,ref_speckle4,sam_speckle1,sam_speckle2,sam_speckle3,sam_speckle4 = get_speckles(para,ref_pha_gt,ref_amp_gt,sam_pha_gt,sam_amp_gt,result_folder)

    # Parameters
    dx = 2e-6
    dy = 2e-6
    lambd = 532e-9

    # 四个相机焦平面的传播距离
    # zs = [para.dist,para.dist2,para.dist3,para.dist4]
    zs = [para.dist]
    # 四个传播距离下相机焦平面的散斑
    ref_speckles = [ref_speckle1,ref_speckle2,ref_speckle3,ref_speckle4]
    
    target_intensity = ref_amp_gt
    rand_phi = np.zeros_like(ref_amp_gt)
    target = target_intensity * np.exp(1j * rand_phi)
    angle_target_list = [rand_phi,rand_phi,rand_phi,rand_phi]

    source_intensitys = ref_speckles

    print(f"target_intensity type: {type(target_intensity)}, dtype: {getattr(target_intensity, 'dtype', 'Not an array')}")
        # 记录开始时间
    refstart_time = time.time()
    ref_pha_pred = targetToSource(target,target_intensity,angle_target_list,zs,source_intensitys,f'{img_txt_folder}/ref',epochs=300)
    # 记录结束时间
    refend_time = time.time()

    # 四个传播距离下相机焦平面的散斑
    sam_speckles = [sam_speckle1,sam_speckle2,sam_speckle3,sam_speckle4]
    source_intensitys = sam_speckles

    target_intensity = ref_amp_gt
    rand_phi = np.zeros_like(ref_amp_gt)
    target = target_intensity * np.exp(1j * rand_phi)
    angle_target_list = [rand_phi,rand_phi,rand_phi,rand_phi]
    # 记录开始时间
    samstart_time = time.time()
    sam_pha_pred = targetToSource(target,target_intensity,angle_target_list,zs,source_intensitys,f'{img_txt_folder}/sam',epochs=300)
    # 记录结束时间
    samend_time = time.time()
    sam_ref_2pi_plot(sam_pha_pred,ref_pha_pred,f'{img_txt_folder}')


    # 计算运行时间
    refelapsed_time = refstart_time - refend_time
    print(f"ref程序运行时间：{refelapsed_time:.2f} 秒")    

    # 计算运行时间
    samelapsed_time = samstart_time - samend_time
    print(f"sam程序运行时间：{samelapsed_time:.2f} 秒")    

    runingtime = open(f"{result_folder}/runingtime.txt","w")
    # 记录训练开始前的超参数，网络结构，输入强度图，gt图像
    runingtime.write(f'ref程序运行时间：{refelapsed_time:.2f} 秒\n')
    runingtime.write(f'sam程序运行时间：{samelapsed_time:.2f} 秒\n')
    runingtime.write(f'总程序运行时间：{refelapsed_time+samelapsed_time:.2f} 秒\n')
    
    
    runingtime.close() 

        




    




  

                

  










