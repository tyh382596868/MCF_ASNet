import torch
import sys
code_path =  '/ailab/user/tangyuhang/LenslessFiberEndomicroscopicPhaseImaging/ASNet'
sys.path.append(code_path)
from config.parameter import import_class
from torch.cuda.amp import autocast as autocast
from tqdm import tqdm
from prop import propcomplex
import numpy as np
import matplotlib.pyplot as plt
from library import my_saveimage,my_savetxt
from utils.compute_metric import compute_std2_plot
from torch.utils.tensorboard import SummaryWriter
# from option import args   
def train_init(result_folder,para):   

    # 随机种子和实验设备
    torch.manual_seed(para.seed)
    device = torch.device(para.device) #分布式训练时用默认

    # 3.model
    print(para.model['name'])

    if para.model['name']=="EDRN":
        modelnet = import_class('arch.'+para.model['filename'],para.model['classname']) 
        net  =  modelnet(args).to(device) 
    elif para.model['name']=="SwinIR":
        print(para.image_height, para.image_width,para.depths)
        modelnet = import_class('arch.'+para.model['filename'],para.model['classname']) 
        net  =  modelnet(upscale=1, img_size=(para.image_height, para.image_width),
                   window_size=para.window_size, img_range=1., depths=para.depths,
                   embed_dim=60, num_heads=[6, 6, 6, 6], mlp_ratio=2, upsampler='pixelshuffledirect').to(device) 
    elif para.model['name']=="phynet":
        print(para.model['name'])
        modelnet = import_class('arch.'+para.model['filename'],para.model['classname']) 
        net  =  modelnet().to(device) 

    elif para.model['name']=="U_Net":
        modelnet = import_class('arch.'+para.model['filename'],para.model['classname']) 
        net  =  modelnet(para.input_channel).to(device) 
    else:
        modelnet = import_class('arch.'+para.model['filename'],para.model['classname']) 
        net  =  modelnet().to(device) 

    print('creating model')

    # 4.loss and optimization
    if para.loss['name'] ==  'MSELoss':
        loss_mse = torch.nn.MSELoss()

    elif para.loss['name'] ==  'L1Loss':
        loss_mse = torch.nn.L1Loss()

    else:
        assert False, "未支持的损失函数类型。只支持 'MSELoss' 和 'L1Loss'。"

    optimizer = torch.optim.Adam(net.parameters(), lr = para.lr)
    print('creating loss and optimization')   
                     
    hypar_file = open(f"{result_folder}/hypar.txt","w")
    # 记录训练开始前的超参数，网络结构，输入强度图，gt图像
    hypar_file.write(f'target： {para.target}\n')
    hypar_file.write(f'para.fi:{para.fi}\n')
    hypar_file.write(f'num of core {para.num}\n')
    hypar_file.write(f'scale of angle {para.scale}\n')
    hypar_file.write(f'dist of prop {para.dist}\n')
    hypar_file.write(f'batch_size {para.batch_size}\n')
    hypar_file.write(f'lr {para.lr}\n')
    hypar_file.write(f'epochs {para.epochs}\n')
    hypar_file.write(f'epochs {para.noise["flag"]},mean {para.noise["mean"]},std {para.noise["std"]}\n')
    hypar_file.write(f'network\n{net}\n')
    
    hypar_file.close() 

    return net,loss_mse,optimizer

def exp_recoder(step,writer,loss_value,pha_gt,flattened_pred_pha,best_loss,net,weight_folder,epochs,img_txt_folder,pred_Speckle,Speckle,amp_gt,mask_gt,std_list,interval=300):
    # 记录loss
    if step % 50 == 0:
        # tb记录loss
        writer.add_scalar('training loss',
                        loss_value.item(),
                        step)
        
        phase_diff = np.abs(pha_gt-((flattened_pred_pha)))

        writer.add_scalar('相位差',
                        np.mean(phase_diff),
                        step)
        # 记录最好的模型权重
        # 保存loss值最小的网络参数
        if loss_value < best_loss:
            best_loss = loss_value
            torch.save(net.state_dict(), f"{weight_folder}/best_model.pth")

    # 记录中间结果图片
    if step % interval == 0 or step+1 == epochs:
                dpi = 800
                my_saveimage(np.mod(flattened_pred_pha,2*np.pi),f'{img_txt_folder}/{step}_PredPhawrap2pi.png',dpi=dpi)
                my_savetxt(np.mod(flattened_pred_pha,2*np.pi),f'{img_txt_folder}/{step}_PredPhawrap2pi.txt')

                my_saveimage((flattened_pred_pha),f'{img_txt_folder}/{step}_PredPha.png',dpi=dpi)
                my_savetxt((flattened_pred_pha),f'{img_txt_folder}/{step}_PredPha.txt')

                my_saveimage(pred_Speckle,f'{img_txt_folder}/{step}_PredAmp.png',dpi=dpi)
                my_savetxt(pred_Speckle,f'{img_txt_folder}/{step}_PredAmp.txt')

                my_saveimage((((Speckle-pred_Speckle))),f'{img_txt_folder}/{step}_AmpLoss.png',dpi=dpi)
                my_savetxt(((Speckle-pred_Speckle)),f'{img_txt_folder}/{step}_AmpLoss.txt')
                
                my_saveimage(np.mod(pha_gt-((flattened_pred_pha)),2*np.pi),f'{img_txt_folder}/{step}_PhaLoss.png',dpi=dpi)
                my_savetxt(np.mod(pha_gt-((flattened_pred_pha)),2*np.pi),f'{img_txt_folder}/{step}_PhaLoss.txt')
                
                # std_phase = compute_std2_plot(mask_gt,np.mod(pha_gt-((flattened_pred_pha)),6.28),f'{img_txt_folder}/{step}core_std.png')
                # std_list.append(std_phase)
                # compute_core_std_plot(amp_gt.cpu().detach().numpy(),np.mod(flattened_pred_pha.cpu().detach().numpy(),2*np.pi),f'{img_txt_folder}/{step}core_std.png',outputflag=True)
                my_saveimage(np.mod((mask_gt*flattened_pred_pha),2*np.pi),f'{img_txt_folder}/{step}_Phamulmask.png',dpi=dpi)
                plt.close()

    if step % 40 == 0:
        # 80的时候显存差不多满了
        torch.cuda.empty_cache()
    # 记录中间结果图片

        
    if step+1 == epochs:
        plt.clf()  # 清图。
        plt.cla()  # 清坐标轴
        plt.figure(figsize=(12, 6))  # 设定图像大小

        # 显示第一个图像
        plt.subplot(2, 2, 1)
        imgplot1 = plt.imshow(np.mod(flattened_pred_pha,2*np.pi), cmap='viridis')
        plt.colorbar()  # 为第一个图像添加颜色条

        # 显示第二个图像
        plt.subplot(2, 2, 2)
        imgplot2 = plt.imshow(pred_Speckle, cmap='viridis')
        plt.colorbar()  # 为第二个图像添加颜色条

        # 显示第一个图像
        plt.subplot(2, 2, 3)
        imgplot1 = plt.imshow(np.mod(pha_gt-((flattened_pred_pha)),2*np.pi), cmap='viridis')
        plt.colorbar()  # 为第一个图像添加颜色条

        # 显示第二个图像
        plt.subplot(2, 2, 4)
        imgplot2 = plt.imshow((Speckle-pred_Speckle), cmap='viridis')
        plt.colorbar()  # 为第二个图像添加颜色条

        plt.savefig(f'{img_txt_folder}/{step}_result.png',dpi=800)  # 保存图像 
        
        plt.clf()  # 清图。
        plt.cla()  # 清坐标轴
        plt.figure(figsize=(12, 6))  # 设定图像大小
        # 绘制方差随迭代次数的变化曲线
        plt.plot(range(1, len(std_list) + 1), std_list)
        plt.xlabel('Iteration')
        plt.ylabel('Standard Deviation')
        plt.title('Standard Deviation over Iterations')
        plt.grid(True)
        plt.savefig(f'{img_txt_folder}/std_list.png',dpi=400)  # 保存图像 

class Trainer():
    def __init__(self,para,datas,result_folder):
        self.para = para

        self.my_model,self.loss_mse,self.optimizer = train_init(result_folder,para)
        self.datas = datas
        self.result_folder = result_folder

        # 最好的loss初始化为无穷大
        self.best_loss = float('inf')


    def train_MultiDist(self):

        # 保存每次计算的方差
        std_list = []
        writer = SummaryWriter(f'{self.result_folder}/tb_folder')
        device = torch.device(self.para.device) #分布式训练时用默认
        pha_gt = torch.tensor(self.datas["pha_gt"]).to(device)
        amp_gt = torch.tensor(self.datas["amp_gt"]).to(device)
        mask_gt = torch.tensor(self.datas["mask_gt"]).to(device)
        speckle_gt = torch.tensor(self.datas["speckle_gt"]).to(device)
        speckle_gt2 = torch.tensor(self.datas["speckle_gt2"]).to(device)
        speckle_gt3 = torch.tensor(self.datas["speckle_gt3"]).to(device)
        speckle_gt4 = torch.tensor(self.datas["speckle_gt4"]).to(device)
        print('starting loop')
        scaler = torch.cuda.amp.GradScaler()

        input = speckle_gt[None,None,:,:]
        print(f"max of input : {torch.max(input)}")

        for current_epoch in tqdm(range(self.para.epochs)):
            
            self.optimizer.zero_grad()
            # forward proapation
            with torch.cuda.amp.autocast():
                
                pred_pha = self.my_model(input) 
                
                flattened_pred_pha = pred_pha[0, 0, :, :] 

                Uo = amp_gt*torch.exp(1j*flattened_pred_pha) #光纤端面初始复光场
                if self.para.NumOfDist == 4:                    
                    Ui = propcomplex(Uo,dist=self.para.dist,device=device)                               
                    pred_Speckle = torch.abs(Ui)  

                    Ui2 = propcomplex(Uo,dist=self.para.dist2,device=device)                               
                    pred_Speckle2 = torch.abs(Ui2)                  

                    Ui3 = propcomplex(Uo,dist=self.para.dist3,device=device)                               
                    pred_Speckle3 = torch.abs(Ui3)

                    Ui4 = propcomplex(Uo,dist=self.para.dist4,device=device)                               
                    pred_Speckle4 = torch.abs(Ui4)

                    loss_mse_value = self.loss_mse(speckle_gt.float(),pred_Speckle.float()) 
                    reg_term = self.loss_mse(speckle_gt2.float(),pred_Speckle2.float()) 
                    reg_term3 = self.loss_mse(speckle_gt3.float(),pred_Speckle3.float())
                    reg_term4 = self.loss_mse(speckle_gt4.float(),pred_Speckle4.float())
                    loss_value =  loss_mse_value+reg_term+reg_term3+reg_term4
                elif self.para.NumOfDist == 3:
                    Ui = propcomplex(Uo,dist=self.para.dist,device=device)                               
                    pred_Speckle = torch.abs(Ui)  

                    Ui2 = propcomplex(Uo,dist=self.para.dist2,device=device)                               
                    pred_Speckle2 = torch.abs(Ui2)                  

                    Ui3 = propcomplex(Uo,dist=self.para.dist3,device=device)                               
                    pred_Speckle3 = torch.abs(Ui3)

                    loss_mse_value = self.loss_mse(speckle_gt.float(),pred_Speckle.float()) 
                    reg_term = self.loss_mse(speckle_gt2.float(),pred_Speckle2.float()) 
                    reg_term3 = self.loss_mse(speckle_gt3.float(),pred_Speckle3.float())

                    loss_value =  loss_mse_value+reg_term+reg_term3                  
                elif self.para.NumOfDist == 2:
                    print(f"NumOfDist:{self.para.NumOfDist}")
                    Ui = propcomplex(Uo,dist=self.para.dist,device=device)                               
                    pred_Speckle = torch.abs(Ui)  

                    Ui2 = propcomplex(Uo,dist=self.para.dist2,device=device)                               
                    pred_Speckle2 = torch.abs(Ui2)                  

                    loss_mse_value = self.loss_mse(speckle_gt.float(),pred_Speckle.float()) 
                    reg_term = self.loss_mse(speckle_gt2.float(),pred_Speckle2.float()) 
                    loss_value =  loss_mse_value+reg_term

                elif self.para.NumOfDist == 1:
                    print(f"NumOfDist:{self.para.NumOfDist}")
                    Ui = propcomplex(Uo,dist=self.para.dist,device=device)                               
                    pred_Speckle = torch.abs(Ui)                  

                    loss_mse_value = self.loss_mse(speckle_gt.float(),pred_Speckle.float()) 

                    loss_value =  loss_mse_value

                # backward proapation 
                # loss_value.backward() 
                scaler.scale(loss_value).backward() 
                # optimizer.step()  
                scaler.step(self.optimizer) 
                scaler.update() 

            exp_recoder(current_epoch,writer,loss_value,pha_gt.cpu().detach().numpy(),flattened_pred_pha.cpu().detach().numpy(),self.best_loss,self.my_model,f"{self.result_folder}/weight_folder",self.para.epochs,f'{self.result_folder}/img_txt_folder',pred_Speckle.cpu().detach().numpy(),speckle_gt.cpu().detach().numpy(),amp_gt.cpu().detach().numpy(),mask_gt.cpu().detach().numpy(),std_list,interval=300)
    
        return flattened_pred_pha

    def train(self):

        # 保存每次计算的方差
        std_list = []
        writer = SummaryWriter(f'{self.result_folder}/tb_folder')
        device = torch.device(self.para.device) #分布式训练时用默认
        pha_gt = torch.tensor(self.datas["pha_gt"]).to(device)
        amp_gt = torch.tensor(self.datas["amp_gt"]).to(device)
        mask_gt = torch.tensor(self.datas["mask_gt"]).to(device)
        speckle_gt = torch.tensor(self.datas["speckle_gt"]).to(device)
        speckle_gt2 = torch.tensor(self.datas["speckle_gt2"]).to(device)
        speckle_gt3 = torch.tensor(self.datas["speckle_gt3"]).to(device)
        speckle_gt4 = torch.tensor(self.datas["speckle_gt4"]).to(device)
        print('starting loop')
        scaler = torch.cuda.amp.GradScaler()

        input = speckle_gt[None,None,:,:]
        print(f"max of input : {torch.max(input)}")

        for current_epoch in tqdm(range(self.para.epochs)):
            
            self.optimizer.zero_grad()
            # forward proapation
            with torch.cuda.amp.autocast():
                
                pred_pha = self.my_model(input) 
                
                flattened_pred_pha = pred_pha[0, 0, :, :] 

                Uo = amp_gt*torch.exp(1j*flattened_pred_pha) #光纤端面初始复光场
                    
                Ui = propcomplex(Uo,dist=self.para.dist,device=device)                               
                pred_Speckle = torch.abs(Ui)  

                Ui2 = propcomplex(Uo,dist=self.para.dist2,device=device)                               
                pred_Speckle2 = torch.abs(Ui2)                  

                Ui3 = propcomplex(Uo,dist=self.para.dist3,device=device)                               
                pred_Speckle3 = torch.abs(Ui3)

                Ui4 = propcomplex(Uo,dist=self.para.dist4,device=device)                               
                pred_Speckle4 = torch.abs(Ui4)

                loss_mse_value = self.loss_mse(speckle_gt.float(),pred_Speckle.float()) 
                reg_term = self.loss_mse(speckle_gt2.float(),pred_Speckle2.float()) 
                reg_term3 = self.loss_mse(speckle_gt3.float(),pred_Speckle3.float())
                reg_term4 = self.loss_mse(speckle_gt4.float(),pred_Speckle4.float())
                loss_value =  loss_mse_value+reg_term+reg_term3+reg_term4
                    


                # backward proapation 
                # loss_value.backward() 
                scaler.scale(loss_value).backward() 
                # optimizer.step()  
                scaler.step(self.optimizer) 
                scaler.update() 

            exp_recoder(current_epoch,writer,loss_value,pha_gt.cpu().detach().numpy(),flattened_pred_pha.cpu().detach().numpy(),self.best_loss,self.my_model,f"{self.result_folder}/weight_folder",self.para.epochs,f'{self.result_folder}/img_txt_folder',pred_Speckle.cpu().detach().numpy(),speckle_gt.cpu().detach().numpy(),amp_gt.cpu().detach().numpy(),mask_gt.cpu().detach().numpy(),std_list,interval=300)
    
        return flattened_pred_pha

    def train2(self):

        # 保存每次计算的方差
        std_list = []
        writer = SummaryWriter(f'{self.result_folder}/tb_folder')
        device = torch.device(self.para.device) #分布式训练时用默认
        pha_gt = torch.tensor(self.datas["pha_gt"]).to(device)
        amp_gt = torch.tensor(self.datas["amp_gt"]).to(device)
        mask_gt = torch.tensor(self.datas["mask_gt"]).to(device)
        speckle_gt = torch.tensor(self.datas["speckle_gt"]).to(device)
        speckle_gt2 = torch.tensor(self.datas["speckle_gt2"]).to(device)
        speckle_gt3 = torch.tensor(self.datas["speckle_gt3"]).to(device)
        speckle_gt4 = torch.tensor(self.datas["speckle_gt4"]).to(device)
        print('starting loop')
        scaler = torch.cuda.amp.GradScaler()

        input = speckle_gt[None,None,:,:]

        for current_epoch in tqdm(range(self.para.epochs)):

            for i in tqdm(range(4)):            
                self.optimizer.zero_grad()
                # forward proapation
                with torch.cuda.amp.autocast():
                    
                    pred_pha = self.my_model(input) 
                    
                    flattened_pred_pha = pred_pha[0, 0, :, :] 

                    Uo = amp_gt*torch.exp(1j*flattened_pred_pha) #光纤端面初始复光场

                    if i==0:                        
                        Ui = propcomplex(Uo,dist=self.para.dist,device=device)                               
                        pred_Speckle = torch.abs(Ui)  
                    elif i==1:
                        Ui = propcomplex(Uo,dist=self.para.dist2,device=device)                               
                        pred_Speckle = torch.abs(Ui)                  
                    elif i==2:
                        Ui = propcomplex(Uo,dist=self.para.dist3,device=device)                               
                        pred_Speckle = torch.abs(Ui)
                    elif i==3:
                        Ui = propcomplex(Uo,dist=self.para.dist4,device=device)                               
                        pred_Speckle = torch.abs(Ui)
                    else:
                        print("*"*20+"wrong")

                    loss_mse_value = self.loss_mse(speckle_gt.float(),pred_Speckle.float()) 
                    loss_value =  loss_mse_value
                        


                    # backward proapation 
                    # loss_value.backward() 
                    scaler.scale(loss_value).backward() 
                    # optimizer.step()  
                    scaler.step(self.optimizer) 
                    scaler.update() 

            exp_recoder(current_epoch,writer,loss_value,pha_gt.cpu().detach().numpy(),flattened_pred_pha.cpu().detach().numpy(),self.best_loss,self.my_model,f"{self.result_folder}/weight_folder",self.para.epochs,f'{self.result_folder}/img_txt_folder',pred_Speckle.cpu().detach().numpy(),speckle_gt.cpu().detach().numpy(),amp_gt.cpu().detach().numpy(),mask_gt.cpu().detach().numpy(),std_list,interval=300)
    
        return flattened_pred_pha

    def train_1dist(self):
        # 保存每次计算的方差
        std_list = []
        writer = SummaryWriter(f'{self.result_folder}/tb_folder')
        device = torch.device(self.para.device) #分布式训练时用默认
        pha_gt = torch.tensor(self.datas["pha_gt"]).to(device)
        amp_gt = torch.tensor(self.datas["amp_gt"]).to(device)
        mask_gt = torch.tensor(self.datas["mask_gt"]).to(device)
        speckle_gt = torch.tensor(self.datas["speckle_gt"]).to(device)
        speckle_gt2 = torch.tensor(self.datas["speckle_gt2"]).to(device)
        speckle_gt3 = torch.tensor(self.datas["speckle_gt3"]).to(device)
        speckle_gt4 = torch.tensor(self.datas["speckle_gt4"]).to(device)
        print('starting loop')
        scaler = torch.cuda.amp.GradScaler()

        input = speckle_gt[None,None,:,:]
        print(f"max of input : {torch.max(input)}")

        for current_epoch in tqdm(range(self.para.epochs)):
            
            self.optimizer.zero_grad()
            # forward proapation
            with torch.cuda.amp.autocast():
                
                pred_pha = self.my_model(input) 
                
                flattened_pred_pha = pred_pha[0, 0, :, :] 

                Uo = amp_gt*torch.exp(1j*flattened_pred_pha) #光纤端面初始复光场
                    
                Ui = propcomplex(Uo,dist=self.para.dist,device=device)                               
                pred_Speckle = torch.abs(Ui)  



                loss_mse_value = self.loss_mse(speckle_gt.float(),pred_Speckle.float()) 

                loss_value =  loss_mse_value
                    


                # backward proapation 
                # loss_value.backward() 
                scaler.scale(loss_value).backward() 
                # optimizer.step()  
                scaler.step(self.optimizer) 
                scaler.update() 

            exp_recoder(current_epoch,writer,loss_value,pha_gt.cpu().detach().numpy(),flattened_pred_pha.cpu().detach().numpy(),self.best_loss,self.my_model,f"{self.result_folder}/weight_folder",self.para.epochs,f'{self.result_folder}/img_txt_folder',pred_Speckle.cpu().detach().numpy(),speckle_gt.cpu().detach().numpy(),amp_gt.cpu().detach().numpy(),mask_gt.cpu().detach().numpy(),std_list,interval=300)
    
        return flattened_pred_pha        