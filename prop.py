import torch


def prop(img,  dx=2e-6, dy=2e-6, lam=532e-9, dist=0.0788,device='cuda:0'):
    '''
    1.注意传播距离，确保生成图像的传播距离与训练的传播距离一致，一般默认一致
    2.输入数据的维度应该为二维，如果是三维，就算值一样傅里叶变换后也不一样
    '''

    # print(torch.max(img))

    img_phase = img #*torch.pi

    # print(torch.max(img_phase))
    
    H = torch.exp(1j * img_phase) 
    fft_H = torch.fft.fftshift(torch.fft.fft2(H))

    (Ny,Nx) = H.shape[0],H.shape[1]
    
    qx = torch.range(1-Nx/2, Nx/2, 1).to(device)
    qy = torch.range(1-Ny/2, Ny/2, 1).to(device)
    y, x = torch.meshgrid(qy, qx)
    r=(2*torch.pi*x/(dx*Nx))**2+(2*torch.pi*y/(dy*Ny))**2

    k=2*torch.pi/lam

    kernel=torch.exp(-1*1j*torch.sqrt(k**2-r)*dist)

    fft_HH=fft_H[:,:]*kernel
    fft_HH=torch.fft.fftshift(fft_HH)

    Ud=torch.fft.ifft2(fft_HH)

    Id=Ud
    Id1=torch.angle(Ud)
    intensity = torch.abs(Id) * torch.abs(Id)
 
    # print(f'torch.max(intensity){torch.max(intensity)}')
    # print(f'intensity{intensity}')

    return intensity

def propcomplex(H, dx=2e-6, dy=2e-6, lam=532e-9, dist=0.0788,device='cuda:0'):
    """Angular Spectrum Method for Light Field Propagation Function

    Args:
        H (Complex Light Field): Input field (for example, a simple plane wave)
        dx (_type_, optional): Spatial sampling interval (meters). Defaults to 2.2e-6.
        dy (_type_, optional): Spatial sampling interval (meters). Defaults to 2.2e-6.
        lam (_type_, optional): Wavelength (meters). Defaults to 532e-9.
        dist (float, optional): Propagation distance (meters). Defaults to 0.0788.
        device (str, optional): GPU or CPU device. Defaults to 'cuda:0'.

    Returns:
        Ud: Complex Light Field

    1.注意传播距离，确保生成图像的传播距离与训练的传播距离一致，一般默认一致
    2.输入数据的维度应该为二维，如果是三维，就算值一样傅里叶变换后也不一样
    """    

    fft_H = torch.fft.fftshift(torch.fft.fft2(H))
    (Ny,Nx) = H.shape[0],H.shape[1]
    
    qx = torch.range(1-Nx/2, Nx/2, 1).to(device)
    qy = torch.range(1-Ny/2, Ny/2, 1).to(device)
    y, x = torch.meshgrid(qy, qx)
    r=(2*torch.pi*x/(dx*Nx))**2+(2*torch.pi*y/(dy*Ny))**2


    k=2*torch.pi/lam

    kernel=torch.exp(-1*1j*torch.sqrt(k**2-r)*dist)

    fft_HH=fft_H[:,:]*kernel
    fft_HH=torch.fft.ifftshift(fft_HH)

    Ud=torch.fft.ifft2(fft_HH)



    return Ud