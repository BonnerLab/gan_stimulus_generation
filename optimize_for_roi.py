from argparse import ArgumentParser
import os
import shutil
import json
from tqdm import tqdm
import torch
from torch import nn
from torchvision.transforms.functional import to_pil_image
from DeePSiM import DeePSiM
from optimize import optimize


def generate_samples(save_path, generator, encoder, target, loss_func, n_samples):
    mean_loss = 0
    for i in range(n_samples):
        generated_image, _, loss = optimize(generator, encoder, target, loss_func)
        generated_image = to_pil_image(generated_image)
        generated_image.save('{}_{}.png'.format(save_path, i))
        mean_loss += loss / n_samples
    with open('{}_metrics.json'.format(save_path), 'w') as f:
        f.write(json.dumps({'mean_final_loss': mean_loss}, indent=2))


if __name__ == '__main__':
    parser = ArgumentParser(description='Optimize images to maximize a class probability using a GAN')
    parser.add_argument('--save_folder', required=True, type=str, help='path to folder to save generated images')
    parser.add_argument('--encoder_file', required=True, type=str, help='name of the encoder file')
    parser.add_argument('--targets_folder', default=None, type=str, help='path to folder containing voxel targets_ppa')
    parser.add_argument('--n_samples', default=10, type=int, help='number of samples to generate per target')
    args = parser.parse_args()

    shutil.rmtree(args.save_folder, ignore_errors=True)
    os.mkdir(args.save_folder)

    encoder = torch.load(os.path.join('saved_models', args.encoder_file),
                         map_location=lambda storage, loc: storage)
    generator = DeePSiM()
    if torch.cuda.is_available():
        encoder.cuda()
        generator.cuda()

    print('Generating targeted stimuli')
    loss_func = nn.MSELoss(reduction='sum')
    targets = os.listdir(args.targets_folder)
    targets = [t for t in targets if t != '.DS_Store']
    targets = [t for t in targets if '.pth' in t]
    for target_name in tqdm(targets):
        target = torch.load(os.path.join(args.targets_folder, target_name))
        save_path = os.path.join(args.save_folder, target_name.split('.')[0])
        generate_samples(save_path, generator, encoder, target, loss_func, args.n_samples)
