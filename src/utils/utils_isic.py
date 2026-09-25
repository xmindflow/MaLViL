import os
import torch

import pandas as pd
import matplotlib.pyplot as plt
from medpy.metric import dc, hd95
import datetime
import cv2
import numpy as np


def histogram_equalization_rgb(image: np.ndarray) -> np.ndarray:
    # Convert to YCrCb color space
    ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    
    # Equalize the Y channel
    ycrcb[..., 0] = cv2.equalizeHist(ycrcb[..., 0])
    
    # Convert back to RGB
    equalized_img = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
    
    return equalized_img



def save_im_gt_pd(im, gt, pd, label, save_path="./results/vis/skin"):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    im = im[0, :].detach().cpu().numpy().transpose(1, 2, 0)
    gt = gt[0, 0].detach().cpu().numpy()
    pd = pd[0, 1].detach().cpu().numpy()
    plt.figure(figsize=(15, 5))
    plt.subplot(131)
    plt.imshow(im[:, :, :3])
    plt.title("Image")
    plt.axis("off")
    plt.subplot(132)
    plt.imshow(gt, cmap="jet")
    plt.title("Ground Truth")
    plt.axis("off")
    plt.subplot(133)
    plt.imshow(pd, cmap="jet")
    plt.title("Prediction")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"{label}.jpeg"))
    plt.close()


def val(net, vl_loader, logging, best_dcs, epoch=0):
    logging.info("Validation ===>")
    dc_sum = 0
    net.eval()
    for i, val_sampled_batch in enumerate(vl_loader):
        val_image_batch, val_label_batch = val_sampled_batch["image"], val_sampled_batch["label"]
        val_image_batch, val_label_batch = val_image_batch.cuda(), val_label_batch.cuda()
        val_outputs = net(val_image_batch)
        val_outputs_binary = torch.argmax(torch.softmax(val_outputs, dim=1), dim=1).squeeze(0)
        dc_sum += dc(val_outputs_binary.detach().cpu().numpy(), val_label_batch[:].detach().cpu().numpy())
    performance = dc_sum / len(vl_loader)

    # print(f"Saving vis. results for validation at epoch: {epoch:03d}")
    # save_im_gt_pd(val_image_batch, val_label_batch, val_outputs, f"{epoch:04d}")

    logging.info('performance in val model) mean_dice:%f, best_dice:%f' % (performance, best_dcs))
    return performance


def test(net, te_loader, logging, best_dcs):
    logging.info("Test ===>")
    dc_sum = 0
    net.eval()
    for i, val_sampled_batch in enumerate(te_loader):
        val_image_batch, val_label_batch = val_sampled_batch["image"], val_sampled_batch["label"]
        val_image_batch, val_label_batch = val_image_batch.type(torch.FloatTensor), val_label_batch.type(torch.FloatTensor)
        val_image_batch, val_label_batch = val_image_batch.cuda(), val_label_batch.cuda()
        val_outputs = net(val_image_batch)
        val_outputs = torch.argmax(torch.softmax(val_outputs, dim=1), dim=1).squeeze(0)
        dc_sum += dc(val_outputs.detach().cpu().numpy(), val_label_batch[:].detach().cpu().numpy())
    performance = dc_sum / len(te_loader)
    logging.info('performance in test model) mean_dice:%f, best_dice:%f' % (performance, best_dcs))
    return performance


def plot_result(dice, h, snapshot_path,args):
    dict = {'mean_dice': dice, 'mean_hd95': h}
    df = pd.DataFrame(dict)
    plt.figure(0)
    df['mean_dice'].plot()
    resolution_value = 1200
    plt.title('Mean Dice')
    date_and_time = datetime.datetime.now()
    filename = f'{args.model_name}_' + str(date_and_time)+'dice'+'.png'
    save_mode_path = os.path.join(snapshot_path, filename)
    plt.savefig(save_mode_path, format="png", dpi=resolution_value)
    plt.figure(1)
    df['mean_hd95'].plot()
    plt.title('Mean hd95')
    filename = f'{args.model_name}_' + str(date_and_time)+'hd95'+'.png'
    save_mode_path = os.path.join(snapshot_path, filename)
    #save csv
    filename = f'{args.model_name}_' + str(date_and_time)+'results'+'.csv'
    save_mode_path = os.path.join(snapshot_path, filename)
    df.to_csv(save_mode_path, sep='\t')
