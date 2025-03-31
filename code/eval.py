import os
import torch
from torch.nn.parallel import DataParallel
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm
from utils import AverageMeter, pad_img, val_psnr, val_ssim
from data import ValDataset
from option import opt
from model import SET_AODNet 


def eval(val_loader, network, max_images=1000):
    PSNR = AverageMeter()
    SSIM = AverageMeter()

    torch.cuda.empty_cache()

    network.eval()

    count = 0 

    for batch in tqdm(val_loader, desc='evaluation'):
        if count >= max_images:  
            break

        hazy_img = batch['hazy'].cuda()
        clear_img = batch['clear'].cuda()

        with torch.no_grad():
            H, W = hazy_img.shape[2:]
            hazy_img = pad_img(hazy_img, 4)
            output = network(hazy_img)
            output = output.clamp(0, 1)
            output = output[:, :, :H, :W]
            if True:
                save_image(output, os.path.join('all_outs/AOD-TCL', batch['filename'][0]))

        psnr_tmp = val_psnr(output, clear_img)
        ssim_tmp = val_ssim(output, clear_img).item()
        PSNR.update(psnr_tmp)
        SSIM.update(ssim_tmp)

        count += 1  

    return PSNR.avg, SSIM.avg


if __name__ == '__main__':
    network = SET_AODNet.dehaze_net().cuda()
    val_dataset_dir = 'Yourpath/dataset/HAZE4K/test' 
    val_dataset = ValDataset(os.path.join(val_dataset_dir, 'hazy'), os.path.join(val_dataset_dir, 'clear'))
    val_loader = DataLoader(val_dataset,
                            batch_size=1,
                            shuffle=False,
                            num_workers=4,
                            pin_memory=True)

    # load pre-trained model
    model_path = 'yourpath/saved_model/180.pk'
    ckpt = torch.load(model_path, map_location='cpu')

    state_dict = ckpt['model']
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v  
        else:
            new_state_dict[k] = v

    network.load_state_dict(new_state_dict)


    avg_psnr, avg_ssim = eval(val_loader, network)
    print('Evaluation on {}\nPSNR:{}\nSSIM:{}'.format(opt.dataset, avg_psnr, avg_ssim))