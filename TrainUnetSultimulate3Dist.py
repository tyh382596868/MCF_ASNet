import sys
code_path =  '/ailab/user/tangyuhang/LenslessFiberEndomicroscopicPhaseImaging/ASNet'
sys.path.append(code_path)
import cv2
from config.parameter import Parameter
import argparse
from copy import deepcopy
import time
from library import mkdir,my_savetxt,sam_ref_2pi_plot

from utils.generate_mcf_simulate import get_speckle2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from trainer import train_init,Trainer


def plotresult(matrix,save_path,dpi=400):
    plt.figure(figsize=(6.67,5))
    plt.imshow(matrix, cmap=cm.viridis)
    plt.colorbar()
    # plt.axis('off')  # Hide axes
    plt.savefig(f'{save_path}.png',dpi=dpi) 
    my_savetxt(matrix,f'{save_path}.txt')

def get_RefSamData(para):

    ref_pha_gt = np.loadtxt('/ailab/user/tangyuhang/ws/Traindata/simulate_data/2pi/15000/5/None/1920-2560/15000_pha_simulate.txt',dtype=np.float32,delimiter=',')
    ref_amp_gt = np.loadtxt('/ailab/user/tangyuhang/ws/Traindata/simulate_data/2pi/15000/5/None/1920-2560/15000_amp_simulate.txt',dtype=np.float32,delimiter=',')    

    # 2.光纤掩膜
    mask = deepcopy(ref_pha_gt)
    mask[mask < 0.02] = 0
    mask[mask > 0.0001] = 1
    kernel = np.ones((11,11),np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # get the phase of sample
    if para.WhichData == "USAF":
        diff_pha_gt = np.loadtxt('/ailab/user/tangyuhang/ws/Traindata/simulate_sample/USAF/simulate_USAF.txt',dtype=np.float32,delimiter=",") 
    elif para.WhichData == "Cell":
        diff_pha_gt = np.loadtxt('/ailab/user/tangyuhang/ws/Traindata/simulate_sample/Cell/Cell.txt',dtype=np.float32,delimiter=",") 
    else:
        raise ValueError("para.WhichData should be 'USAF' or 'Cell'")

    if para.scale == "2pi":
        diff_pha_gt = diff_pha_gt/4*3
    elif para.scale == "pi":
        ref_pha_gt = ref_pha_gt/8
        diff_pha_gt = diff_pha_gt/8
    else:
        raise ValueError("para.WhichData should be '2pi' or 'pi'")

    return ref_pha_gt,ref_amp_gt,mask,diff_pha_gt

def get_RefSamRealData():

    ref_pha_gt = np.loadtxt('/ailab/user/tangyuhang/ws/Traindata/real_data/TestChartRaw/testChartRefPha.txt',dtype=np.float32,delimiter=',')
    ref_amp_gt = np.loadtxt('/ailab/user/tangyuhang/ws/Traindata/real_data/TestChartRaw/testChartRefAmp.txt',dtype=np.float32,delimiter=',')    

    # 2.光纤掩膜
    mask = deepcopy(ref_pha_gt)
    mask[mask < 0.02] = 0
    mask[mask > 0.0001] = 1
    kernel = np.ones((11,11),np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # get the phase of sample
    sam_pha_gt = np.loadtxt('/ailab/user/tangyuhang/ws/Traindata/real_data/TestChartRaw/testChartSamPha.txt',dtype=np.float32,delimiter=',')
    sam_amp_gt = np.loadtxt('/ailab/user/tangyuhang/ws/Traindata/real_data/TestChartRaw/testChartSamAmp.txt',dtype=np.float32,delimiter=',') 

    return ref_pha_gt,ref_amp_gt,mask,sam_pha_gt,sam_amp_gt

if __name__=="__main__":

    parser = argparse.ArgumentParser(description='Training script')
    parser.add_argument('--opt', type=str, default=f'{code_path}/option/Sultimulate3Dist.yaml', help='Path to the configuration file')
    args = parser.parse_args()
    para = Parameter(args.opt)

    localtime = time.strftime("%Y-%m-%d-%H-%M", time.localtime())   

    result_folder = f"/ailab/user/tangyuhang/LenslessFiberEndomicroscopicPhaseImaging/Result/{para.exp_name}_{para.model['name']}/{para.RealOrSimulate}/{para.NumOfDist}/{localtime}"
    # 创建文件夹
    mkdir(result_folder)  
    mkdir(f'{result_folder}/ref')
    mkdir(f'{result_folder}/ref/tb_folder')
    mkdir(f"{result_folder}/ref/weight_folder")
    mkdir(f'{result_folder}/ref/img_txt_folder')

    mkdir(f'{result_folder}/sam')  
    mkdir(f'{result_folder}/sam/tb_folder')
    mkdir(f"{result_folder}/sam/weight_folder")
    mkdir(f'{result_folder}/sam/img_txt_folder')
# load the phase of the fiber distortion

    if para.RealOrSimulate == 'Simulate':
        pha_gt,amp_gt,mask_gt,sample_pha = get_RefSamData(para)

        pha_gt = pha_gt*mask_gt
        amp_gt = amp_gt*mask_gt     

    # load the phase of the sample distortion
        sam_pha_gt = np.mod(pha_gt+sample_pha,2*np.pi)    
        sam_amp_gt = amp_gt

        sam_pha_gt = sam_pha_gt*mask_gt
        sam_amp_gt = sam_amp_gt*mask_gt   
        print(np.max(sample_pha))
        plotresult(sample_pha,f'{result_folder}/sample_pha')

    elif para.RealOrSimulate == 'Real':
        pha_gt,amp_gt,mask_gt,sam_pha_gt,sam_amp_gt = get_RefSamRealData()
        
    else:
        raise ValueError("RealOrSimulate 必须是Real Or Simulate")

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

    ref_datas = {"pha_gt":pha_gt,"amp_gt":amp_gt,"mask_gt":mask_gt,"speckle_gt":speckle_gt,"speckle_gt2":speckle_gt2,"speckle_gt3":speckle_gt3,"speckle_gt4":speckle_gt4}
    sam_datas = {"pha_gt":sam_pha_gt,"amp_gt":sam_amp_gt,"mask_gt":mask_gt,"speckle_gt":sam_speckle_gt,"speckle_gt2":sam_speckle_gt2,"speckle_gt3":sam_speckle_gt3,"speckle_gt4":sam_speckle_gt4}
    

    t = Trainer(para,ref_datas,f"{result_folder}/ref")
    pred_pha = t.train_MultiDist()

    tSam = Trainer(para,sam_datas,f"{result_folder}/sam")
    sam_pred_pha = tSam.train_MultiDist()    
    sam_ref_2pi_plot(pred_pha.cpu().detach().numpy(),sam_pred_pha.cpu().detach().numpy(),result_folder)










  

                

  










