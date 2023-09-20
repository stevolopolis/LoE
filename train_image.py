import hydra
from easydict import EasyDict

import torch
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim_func
from skimage.metrics import peak_signal_noise_ratio as psnr_func

from loe import LoE
from dataset import ImageFileDataset


def train(configs, model, dataset, device='cuda'):
    train_configs = configs.TRAIN_CONFIGS
    model_configs = configs.model_config

    opt = torch.optim.Adam(model.parameters(), lr=train_configs.lr)

    # prepare training settings
    model.train()
    model = model.to(device)
    process_bar = tqdm(range(train_configs.iterations))
    H, W, C = dataset.H, dataset.W, dataset.C

    coords, labels = dataset.get_data()
    coords, labels = coords.to(device), labels.to(device)
    ori_img = labels.view(H, W, C).cpu().detach().numpy()
    ori_img = (ori_img + 1) / 2 if model_configs.INPUT_OUTPUT.data_range == 2 else ori_img

    # train
    for step in process_bar:
        preds = model(coords, labels)
        loss = ((preds - labels) ** 2).mean()       # MSE loss

        # backprop
        opt.zero_grad()
        loss.backward()
        opt.step()

        preds = preds.clamp(-1, 1).view(H, W, C)       # clip to [-1, 1]
        preds = (preds + 1) / 2                        # [-1, 1] -> [0, 1]

        preds = preds.cpu().detach().numpy()
        psnr_score = psnr_func(preds, ori_img, data_range=1)
        ssim_score = ssim_func(preds, ori_img, channel_axis=-1, data_range=1)
        
        # udpate progress bar
        process_bar.set_description(f"psnr: {psnr_score:.2f}, ssim: {ssim_score*100:.2f}, loss: {loss.item():.4f}")
        
    print("Training finished!")


@hydra.main(version_base=None, config_path='config', config_name='train_image')
def main(configs):
    configs = EasyDict(configs)

    dataset = ImageFileDataset(configs.DATASET_CONFIGS, configs.model_config.INPUT_OUTPUT)
    
    model = LoE(
        in_features=dataset.dim_in, 
        out_features=dataset.dim_out,
        loe_configs=configs.model_config
    )
    print(f"Start experiment: {configs.TRAIN_CONFIGS.out_dir}")
    n_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
    print(f"No. of parameters: {n_params}")

    # train
    psnr, ssim = train(configs, model, dataset, device=configs.TRAIN_CONFIGS.device)

    return psnr, ssim, n_params

if __name__=='__main__':
    main()