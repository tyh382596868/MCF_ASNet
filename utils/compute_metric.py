import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mode

from scipy import ndimage
def compute_std(mask,pha_diff):
    """
    Compute the standard deviation of the phase difference between the masked region and the background.
    """
    pha = mask * pha_diff
    pha = np.mod(pha,2*np.pi)

    
    non_zero_elements = pha[pha != 0]
    mean = np.mean(non_zero_elements)
    std = np.std(non_zero_elements)
    print(mean,std)

    
    return std

def compute_std2(mask,pha_diff):
    """
    Compute the standard deviation of the phase difference between the masked region and the background.
    """
    pha = mask * pha_diff
    pha = np.mod(pha,2*np.pi)

    
    # 提取圆内的相位值
    phase_values = pha_diff[mask.astype(bool)]

    # 计算平均值和方差
    mean_phase = np.mean(phase_values)
    std_phase = np.std(phase_values)
    print(mean_phase,std_phase)



    
    return std_phase,mean_phase,phase_values

def compute_std2_plot(mask,pha_diff,save_path):
    """
    Compute the standard deviation of the phase difference between the masked region and the background.
    """
    std_phase,mean_phase,phase_values = compute_std2(mask,pha_diff)

    sampled_indices = np.random.choice(len(phase_values), size=15000, replace=False)
    sampled_phase_values = phase_values[sampled_indices]

    # 绘制散点图
    plt.figure()
    plt.scatter(range(len(sampled_phase_values)), sampled_phase_values,marker='o',s=10,c='r', label='Phase Values')
    plt.axhline(y=mean_phase, color='r', linestyle='-', label=f'Mean: {mean_phase:.2f}')
    plt.axhline(y=mean_phase + std_phase, color='g', linestyle='--', label=f'Standard Deviation: {std_phase:.2f}')
    plt.axhline(y=mean_phase - std_phase, color='g', linestyle='--')
    # plt.title(f'Iteration {iteration + 1}')
    plt.xlabel('Index')
    plt.ylabel('Phase Value')
    plt.legend()
    plt.grid(True)
    # 保存图片
    plt.savefig(save_path)

    
    return std_phase

def compute_core_std(mask,pha_diff):
    '''
    Compute the standard deviation of the phase difference between the masked region and the background.
    
    '''
    pha = mask * pha_diff
    print(type(pha))
    pha = np.mod(pha,2*np.pi)
    
    scale = (pha > 0.01) & (pha < (2*np.pi-0.01))
    
    labeled_array, num_features = ndimage.label(scale)
    
    # plt.figure(figsize=(6, 6))
    # plt.imshow(labeled_array,cmap='viridis')
    # plt.colorbar()
    # plt.show()
    
    # print(num_features)
    
    output = np.zeros_like(pha)
    
    core_mean_values = []
    
    for i in range(1,num_features+1):
        mask_core = (labeled_array == i)
        core_mean_value = pha[mask_core].mean()
        core_mean_values.append(core_mean_value)
        output[mask_core] = core_mean_value
        
    mean = np.mean(core_mean_values)
    std = np.std(core_mean_values)    
    # print(mean_values)
    # plt.figure(figsize=(6, 6))
    # plt.imshow(output,cmap='viridis')
    # plt.colorbar()
    # plt.show()
    
    # plt.figure(figsize=(8,6))
    # plt.scatter(range(len(core_mean_values)),core_mean_values,marker='o',s=10,c='r')
    
    
    # ax = plt.gca()
    # ax.set_yticks(np.arange(0,2*np.pi,0.4))
    # plt.title('Phase difference distribution')
    # plt.xlabel('Index')
    # plt.ylabel('Phase difference')
    # plt.grid(True)
    
    # plt.axhline(y=mean,color='g',linestyle='-',label=f'Mean:{mean:.4f}')
    # plt.axhline(y=mean+std,color='b',linestyle='--',label=f'Mean+std:{mean+std:.4f}')
    # plt.axhline(y=mean-std,color='b',linestyle='--',label=f'Mean-std:{mean-std:.4f}')
    # plt.axhline(y=std,color='black',linestyle='--',label=f'std:{std:.4f}')
    # plt.legend(loc='lower left')
    # plt.show()    
    return core_mean_values,mean,std,output,num_features,labeled_array
def compute_core_std_plot(mask,pha_diff,save_path,meanflag=None,outputflag=None,labeledflag=None):
    '''
    Compute the standard deviation of the phase difference between the masked region and the background.
    
    '''
    # 设置字体为Arial
    # plt.rcParams['font.sans-serif'] = ['Arial']

    # 调整字体大小
    plt.rcParams.update({'font.size': 14})
    core_mean_values,mean,std,output,num_features,labeled_array = compute_core_std(mask,pha_diff)
    if meanflag is not None:
        plt.figure(figsize=(6.67,5))
        plt.scatter(range(len(core_mean_values)),core_mean_values,marker='o',s=10,c='r')
        
        ax = plt.gca()
        ax.set_yticks(np.arange(0,2*np.pi,0.4))
        # plt.yticks(theta, [ '0', 'π/2', 'π', '3π/2', '2π'])
        plt.title('Phase difference distribution')
        plt.xlabel('Index')
        plt.ylabel('Phase difference')
        # plt.grid(True)
        
        plt.axhline(y=mean,color='g',linestyle='-',label=f'Mean:{mean:.4f}')
        plt.axhline(y=mean+std,color='b',linestyle='--',label=f'Mean+std:{mean+std:.4f}')
        plt.axhline(y=mean-std,color='b',linestyle='--',label=f'Mean-std:{mean-std:.4f}')
        plt.axhline(y=std,color='black',linestyle='--',label=f'std:{std:.4f}')
        
        plt.legend(loc='lower left')
        plt.savefig(save_path) 
        
     
    if outputflag is not None:
        plt.figure(figsize=(6.67,5))
        plt.imshow(output,cmap='viridis')
        plt.colorbar() 
        # plt.title(f'num_features:{num_features}')
        plt.savefig(save_path.replace('core_std','_PredPhaclean'))   
        
    if labeledflag is not None:
        plt.figure(figsize=(6.67,5))
        plt.imshow(labeled_array,cmap='viridis')
        plt.colorbar()
        plt.title(f'num_features:{num_features}')
        plt.savefig(save_path.replace('core_std','labeled'))   
    
 
    

if __name__ == '__main__':
    # mask = np.loadtxt('D:\\tyh\simulateData\simulate_data\strong\\2pi\\10\\0.6\\4\\10_amp_simulate.txt',dtype=np.float32,delimiter=',')
    # pha_diff = np.loadtxt('D:\\tyh\Resultimulate\strong\\2pi\\10\\0.6\\4\\2024-04-10-16-50\img_txt_folder\\9000_PhaLoss.txt',dtype=np.float32,delimiter=',')   
    # save_path = 'D:\\tyh\\phynet\\utils\\std\\10.png'
    
    
    # mask = np.loadtxt('D:\\tyh\simulateData\simulate_data\strong\\2pi\\100\\0.6\\4\\100_amp_simulate.txt',dtype=np.float32,delimiter=',')
    # pha_diff = np.loadtxt('D:\\tyh\Resultimulate\strong\\2pi\\100\\0.6\\4\\2024-04-10-16-33\img_txt_folder\\3000_PhaLoss.txt',dtype=np.float32,delimiter=',')   
    # save_path = 'D:\\tyh\\phynet\\utils\\std\\100.png'    
    
    # mask = np.loadtxt('D:\\tyh\simulateData\simulate_data\strong\\2pi\\100\\0.6\\4\\100_amp_simulate.txt',dtype=np.float32,delimiter=',')
    # pha_diff = np.loadtxt('D:\\tyh\Resultimulate\strong\\2pi\\100\\0.6\\4\\2024-04-10-16-33\img_txt_folder\\3000_PhaLoss.txt',dtype=np.float32,delimiter=',')   
    # save_path = 'D:\\tyh\\phynet\\utils\\std\\100.png'  
    
    # mask = np.loadtxt('D:\\tyh\\simulateData\\simulate_data\\strong\\2pi\\100\\0.6\\2\\100_amp_simulate.txt',dtype=np.float32,delimiter=',')
    # pha_pred = np.loadtxt('D:\\tyh\GS\\100_predpha.txt',dtype=np.float32,delimiter=',')   
    # pha_gt = np.loadtxt('D:\\tyh\simulateData\simulate_data\strong\\2pi\\100\\0.6\\2\\100_pha_simulate.txt',dtype=np.float32,delimiter=',')   
    # pha_diff = np.mod(pha_pred-pha_gt,2*np.pi)   
    # save_path = 'D:\\tyh\\phynet\\utils\\std\\100.png'    
    # compute_core_std_plot(mask,pha_diff,save_path,meanflag=True)

    mask = np.loadtxt('/ailab/user/tangyuhang/ws/Traindata/real_data/blood/blood_mask_ref.txt',dtype=np.float32,delimiter=',')
    pha_pred = np.loadtxt('/ailab/user/tangyuhang/ws/Resultimulate/real_Generator_2dist/strong/2pi/2024-08-16-17-32/0.01-0.03-0.05/ref/img_txt_folder/600_PhaLoss.txt',dtype=np.float32,delimiter=',')   
    # pha_pred = pha_pred*mask
    compute_std(mask,pha_pred)
    compute_std2(mask,pha_pred)
   
    # pha_diff = np.mod(pha_pred,2*np.pi)   
    
    # plt.figure(figsize=(6, 6))
    # plt.imshow(pha_diff,cmap='viridis')
    # plt.colorbar()
    # plt.show()
   
