import os
import torch

import pandas as pd
import matplotlib.pyplot as plt
from medpy.metric import dc, hd95
import datetime
import cv2
import numpy as np
from PIL import Image
from utils import compute_segmentation_metrics_hard


def calc_iou(pred, gt):
    """
    Calculate Intersection over Union (IoU) for binary masks.
    
    Args:
        pred (np.ndarray): Predicted binary mask.
        gt (np.ndarray): Ground truth binary mask.
        
    Returns:
        float: IoU score.
    """
    intersection = np.logical_and(pred, gt)
    union = np.logical_or(pred, gt)
    iou_score = np.sum(intersection) / np.sum(union) if np.sum(union) > 0 else 0.0
    return iou_score


def histogram_equalization_rgb(image: np.ndarray) -> np.ndarray:
    # Convert to YCrCb color space
    ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    
    # Equalize the Y channel
    ycrcb[..., 0] = cv2.equalizeHist(ycrcb[..., 0])
    
    # Convert back to RGB
    equalized_img = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
    
    return equalized_img



def save_im_gt_pd_hot(im, gt, pd, label, save_path="../results/vis/skin"):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if isinstance(im, torch.Tensor):
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


def skin_plot(img, gt, pred):
    edged_test = cv2.Canny(pred, 100, 255)
    contours_test, _ = cv2.findContours(edged_test, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    edged_gt = cv2.Canny(gt, 100, 255)
    contours_gt, _ = cv2.findContours(edged_gt, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt_test in contours_test:
        cv2.drawContours(img, [cnt_test], -1, (0, 0, 255), 1)
    for cnt_gt in contours_gt:
        cv2.drawContours(img, [cnt_gt], -1, (0,255,0), 1)
    return img

def save_im_gt_pd(im, gt, pd, fid, save_path="../results/vis/skin"):
        os.makedirs(save_path, exist_ok=True)
        im = (im - im.min()) / (im.max() - im.min() + 1e-8)
        gt = (gt - gt.min()) / (gt.max() - gt.min() + 1e-8)
        pd = (pd - pd.min()) / (pd.max() - pd.min() + 1e-8)
        im = np.ascontiguousarray(im*255., dtype=np.uint8)
        if im.shape[-1] == 1:
            im = im.repeat(3, axis=-1)
        img = im.copy()
        gt = np.uint8(gt*255.)
        pd = np.ascontiguousarray(pd*255., dtype=np.uint8)
        
        res_img = skin_plot(im, gt, pd)
        
        Image.fromarray(img).save(f"{save_path}/{fid}_img.png")
        Image.fromarray(gt).save(f"{save_path}/{fid}_gt.png")
        Image.fromarray(res_img).save(f"{save_path}/{fid}_img_gt_pred.png")



def val(net, vl_loader, logging, best_dcs, epoch=0):
    logging.info("Validation ===>")
    dc_sum = 0
    net.eval()
    for i, batch_data in enumerate(vl_loader):
        batch_image, batch_label = batch_data["image"], batch_data["label"]
        batch_image, batch_label = batch_image.cuda(), batch_label.cuda()
        val_outputs = net(batch_image)
        if isinstance(val_outputs, tuple): # for outputs, rec_error in some models
            val_outputs = val_outputs[0]
        if isinstance(val_outputs, list): # for deep supervision outputs
            val_outputs = val_outputs[-1]
        val_outputs_binary = torch.argmax(torch.softmax(val_outputs, dim=1), dim=1).squeeze(0)
        dc_sum += dc(val_outputs_binary.detach().cpu().numpy(), batch_label[:].detach().cpu().numpy())
    performance = dc_sum / len(vl_loader)

    # print(f"Saving vis. results for validation at epoch: {epoch:03d}")
    # save_im_gt_pd_hot(batch_image, batch_label, val_outputs, f"{epoch:04d}")

    logging.info('performance in val model) mean_dice:%f, best_dice:%f' % (performance, best_dcs))
    return performance



@torch.no_grad()
def test(net, te_loader, logging, best_dcs, save_path=None, ignore_index=None, include_background=False):
    logging.info("Test ===>")
    net.eval()

    # dataset-level accumulators (we'll average per-batch metrics weighted by batch size)
    total_input = 0

    sum_mean_dice = 0.0
    sum_mean_iou = 0.0
    sum_mean_acc = 0.0
    correct_total = 0
    total_pixels = 0
    total_loss = 0
    total_input = 0
    total_dice, total_miou, total_acc, total_hd95, total_jc, total_asd = 0, 0, 0, 0, 0, 0
    pixel_accs = []
    for batch_data in te_loader:
        batch_image = batch_data["image"].float().cuda(non_blocking=True)
        batch_label = batch_data["label"].long().cuda(non_blocking=True)  # [B,H,W]
        batch_id = batch_data.get("id", None)

        outputs = net(batch_image)
        # compute dice, accumulate in total_dice ...
        if isinstance(outputs, tuple):
            outputs, rec_loss = outputs[0], outputs[1]
        if isinstance(outputs, list):
            logits = outputs[-1]
        else:
            logits = outputs
            
        metrics = compute_segmentation_metrics_hard(logits, batch_label, ignore_index=None, include_background=False, eps=1e-8)
        '''
        METRICS:
            - "per_class_dice"
            - "mean_dice"
            - "per_class_iou"
            - "mean_iou"
            - "per_class_accuracy"
            - "mean_accuracy"
            - "per_class_hd95"
            - "mean_hd95"
            - "per_class_jaccard"
            - "mean_jaccard"
            - "per_class_asd"
            - "mean_asd"
            - "num_classes"
            - "ignore_index"
            - "include_background_in_mean"
        '''
        B = batch_label.shape[0]
        total_dice += metrics["mean_dice"] * B
        total_miou += metrics["mean_iou"] * B
        total_acc += metrics["mean_accuracy"] * B
        total_hd95 += metrics["mean_hd95"] * B
        total_jc += metrics["mean_jaccard"] * B
        total_asd += metrics["mean_asd"] * B
        
        # compute pixel-wise accuracy for logging
        pred_binary = torch.argmax(logits, dim=1)  # [B,H,W]
        correct = (pred_binary == batch_label).sum().item()
        total_pixels = batch_label.numel()
        pixel_acc = correct / total_pixels
        pixel_accs.append(pixel_acc * B)  # accumulate weighted by batch size
        
        total_input += B

        # OPTIONAL: save images per sample (keep your existing saver if you want)
        if save_path is not None and batch_id is not None:
            pred = logits.argmax(dim=1)  # [B,H,W]
            for b in range(B):
                im = batch_image[b, :3].detach().cpu().numpy().transpose(1, 2, 0)
                gt = batch_label[b].detach().cpu().numpy()
                pd = pred[b].detach().cpu().numpy()
                name = batch_id[b].item() if not isinstance(batch_id[b], str) else batch_id[b]
                save_im_gt_pd(im, gt, pd, name, save_path)

    avg_dice = sum_mean_dice / max(total_input, 1)
    avg_iou = sum_mean_iou / max(total_input, 1)
    avg_acc = sum_mean_acc / max(total_input, 1)

    avg_loss = total_loss / max(total_input, 1)
    avg_dcs = total_dice / max(total_input, 1)
    avg_miou = total_miou / max(total_input, 1)
    avg_acc = total_acc / max(total_input, 1)
    avg_hd95 = total_hd95 / max(total_input, 1)
    avg_jc = total_jc / max(total_input, 1)
    avg_asd = total_asd / max(total_input, 1)
    avg_pixel_acc = np.mean(pixel_accs)

    return avg_dcs, avg_pixel_acc, avg_miou

# def test(net, te_loader, logging, best_dcs, save_path=None):
#     logging.info("Test ===>")
#     dc_sum = 0
#     acc_sum = 0
#     total_pixels = 0
#     ious = []
#     net.eval()
#     for i, batch_data in enumerate(te_loader):
#         batch_image, batch_label = batch_data["image"], batch_data["label"]
#         batch_id = batch_data["id"]
#         batch_image, batch_label = batch_image.type(torch.FloatTensor), batch_label.type(torch.FloatTensor)
#         batch_image, batch_label = batch_image.cuda(), batch_label.cuda()
#         te_outputs = net(batch_image)
#         if isinstance(te_outputs, tuple): # for outputs, rec_error in some models
#             te_outputs = te_outputs[0]
#         if isinstance(te_outputs, list): # for deep supervision outputs
#             te_outputs = te_outputs[-1]
#         te_outputs = torch.argmax(torch.softmax(te_outputs, dim=1), dim=1).squeeze(0)

#         pd = te_outputs.detach().cpu().numpy()
#         gt = batch_label[0, 0].detach().cpu().numpy()
#         # Accuracy calculation
#         correct = (pd == gt).sum()
#         acc_sum += correct
#         total_pixels += gt.size

#         if save_path is not None:
#             # Save the image, ground truth, and prediction
#             im = batch_image[0, :3].cpu().detach().numpy().transpose(1, 2, 0)
#             id = batch_id[0]
#             save_im_gt_pd(im, gt, pd, id.item(), save_path=save_path)
#         ious.append(calc_iou(pd>0.5, gt>0.5))

#         dc_sum += dc(pd, batch_label[:].detach().cpu().numpy())

#     avg_dice = dc_sum / len(te_loader)
#     avg_iou = np.mean(ious)
#     avg_acc = acc_sum / total_pixels
#     # if best_dcs < 0 or best_dcs is None:
#     #     print('performance in test model) mean_dice:%f, iou:%f, acc:%f' % (avg_dice, avg_iou, avg_acc))
#     # else:
#     #     print('performance in test model) mean_dice:%f, best_dice:%f, iou:%f, acc:%f' % (avg_dice, best_dcs, avg_iou, avg_acc))
#     return avg_dice, avg_acc, avg_iou


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





normalize = lambda x: (x - x.min()) / (x.max() - x.min() + 1e-8)
def binary_save_test_images(net, te_loader, save_imgs_dir, device):
    from PIL import Image
    import cv2
    def skin_plot(img, gt, pred):
        img = np.array(img)
        gt = np.array(gt); gt = np.where(normalize(gt)>0.5, 255, 0).astype(np.uint8)
        pred = np.array(pred); pred = np.where(normalize(pred)>0.5, 255, 0).astype(np.uint8)
        edged_test = cv2.Canny(pred, 100, 255)
        contours_test, _ = cv2.findContours(edged_test, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        edged_gt = cv2.Canny(gt, 100, 255)
        contours_gt, _ = cv2.findContours(edged_gt, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for cnt_test in contours_test:
            cv2.drawContours(img, [cnt_test], -1, (0, 0, 255), 1)
        for cnt_gt in contours_gt:
            cv2.drawContours(img, [cnt_gt], -1, (0,255, 0), 1)
        return img
    #---------------------------------------------------------------------------------------------
    if not os.path.isdir(save_imgs_dir):
        os.mkdir(save_imgs_dir)

    with torch.no_grad():
        for batch in te_loader:
            imgs = batch['image']
            msks = batch['label']
            ids = batch['id']
            
            outputs = net(imgs.to(device))
            if isinstance(outputs, tuple): # for outputs, rec_error in some models
                outputs, rec_error = outputs[0], outputs[1]
            if isinstance(outputs, list): # for deep supervision outputs
                outputs = outputs[-1]

            preds = torch.argmax(outputs, 1).cpu().numpy()
            for idx in range(len(msks)):

                if hasattr(te_loader.dataset, 'make_pil_img'):
                    pil_img = te_loader.dataset.make_pil_img(imgs[idx])
                    img = np.array(pil_img, dtype=np.uint8)
                else:                    
                    img = np.moveaxis(imgs[idx, :3].cpu().numpy(), 0, -1)
                    img = np.ascontiguousarray(img*255., dtype=np.uint8)
                    if img.shape[-1] == 1:
                        img = img.repeat(3, axis=-1)

                gt = np.uint8(msks[idx]*255.)
                pred = np.where(preds[idx]>0.5, 255, 0)
                pred = np.ascontiguousarray(pred, dtype=np.uint8)
                
                res_img = skin_plot(img, gt, pred)
                
                fid = ids[idx]
                Image.fromarray(img).save(f"{save_imgs_dir}/{fid}_img.png")
                Image.fromarray(gt).save(f"{save_imgs_dir}/{fid}_gt.png")
                Image.fromarray(res_img).save(f"{save_imgs_dir}/{fid}_img_gt_pred.png")

    print(f"directory for visualization:\n  {save_imgs_dir}")



