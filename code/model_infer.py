import os
import torch
from torchvision.utils import save_image
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F

from model import SET_AODNet 
from option_train import opt  


output_dir = './outs'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

device = torch.device(opt.device if torch.cuda.is_available() else 'cpu')
net = SET_AODNet.dehaze_net().to(device)
net.eval()


model_path = 'Yourpath/outdoor/1000iter_outdoor/saved_model/xxx.pk'  
checkpoint = torch.load(model_path, map_location=device)

if 'model' in checkpoint:
    net.load_state_dict(checkpoint['model'])
    print("Model weights loaded successfully.")
else:
    print("Error: 'model' key not found in the checkpoint. Please check the .pk file.")
# checkpoint = torch.load(model_path, map_location=device)
# net.load_state_dict(checkpoint['model'])

transform = transforms.Compose([
    transforms.ToTensor(),
])

def pad_img(x, patch_size):
    _, _, h, w = x.size()
    mod_pad_h = (patch_size - h % patch_size) % patch_size
    mod_pad_w = (patch_size - w % patch_size) % patch_size
    x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
    return x

def infer_image(img_path, output_dir):
    img = Image.open(img_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device) 

    H, W = img_tensor.shape[2:]
    img_tensor = pad_img(img_tensor, 4)

    with torch.no_grad():
        output = net(img_tensor).clamp(0, 1)
        output = output[:, :, :H, :W]  

    img_name = os.path.basename(img_path)
    save_image(output, os.path.join(output_dir, f'result_{img_name}'))
    print(f"Saved result to {os.path.join(output_dir, f'result_{img_name}')}")


if __name__ == "__main__":

    img_paths = ''
    
    for img_path in img_paths:
        infer_image(img_path, output_dir)
