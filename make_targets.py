import torch
from utils import image_to_tensor
import os

save_dir = '/home/eric/Desktop/stimulus_generation/coco/targets_ppa'
data_dir = '/home/eric/Desktop/stimulus_generation/coco/stimuli'
encoder_name = 'study=bold5000_featextractor=alexnet_featname=conv_3_rois=PPA.pth'
resolution = 256

encoder = torch.load(os.path.join('saved_models', encoder_name),
                    map_location=lambda storage, loc: storage)
os.mkdir(save_dir)
image_names = os.listdir(data_dir)
images_names = [img for img in image_names if img != '.DS_Store']
for name in images_names:
    path = os.path.join(data_dir, name)
    image = image_to_tensor(path, resolution=resolution)
    target = encoder(image.unsqueeze(dim=0)).squeeze(dim=0)
    torch.save(target, os.path.join(save_dir, name + '.pth'))
