
import torch
import os
import copy
import numpy as np
from utils import test_single_volume, compute_segmentation_metrics_hard
from utils.metrics_eval import select_fast_slice_indices

try:
    from torch.amp import autocast, GradScaler
except ImportError:
    from torch.amp.autocast_mode import autocast
    from torch.amp.grad_scaler import GradScaler


LAMBDA_REC = 0.01  # paper λ on Bi-LRViL reconstruction
# Synapse has no held-out test. Fast validation on the 12 volumes picks best.pth.
# The reported number is a later full-volume pass on that checkpoint.
_SYNAPSE_VAL_AFTER = 50
_SYNAPSE_VAL_INTERVAL = 5
_FAST_VAL_SLICES = 24


def inference_3d(
    args,
    net,
    loader,
    test_save_path=None,
    epoch=0,
    logging=None,
    fast_val=None,
    fast_val_slices=None,
):
    """3D volume inference.

    Mid-run Synapse training uses fast validation: a GT-coverage slice subset.
    Last epoch and `--eval` always score the full volumes of best.pth.
    """
    use_fast = bool(fast_val) if fast_val is not None else False
    if epoch >= getattr(args, "max_epochs", epoch):
        use_fast = False
    n_fast = int(fast_val_slices if fast_val_slices is not None else _FAST_VAL_SLICES)

    mode = f"fast validation slices={n_fast}" if use_fast else "full validation"
    logging.info(f"{len(loader)} validation volumes | mode={mode}")
    net.eval()
    metric_rows = []
    total_loss = 0.0
    with torch.no_grad():
        for i_batch, sampled_batch in enumerate(loader):
            image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]

            slice_indices = None
            if use_fast:
                # GT-aware slice schedule: cover every organ present in the volume.
                slice_indices = select_fast_slice_indices(
                    label,
                    k=n_fast,
                    num_classes=args.num_classes,
                    seed=getattr(args, "seed", 1234) + i_batch,
                )

            pred, metric_i = test_single_volume(
                image,
                label,
                net,
                classes=args.num_classes,
                patch_size=[args.img_size, args.img_size],
                test_save_path=test_save_path,
                case=case_name,
                z_spacing=args.z_spacing,
                epoch=epoch,
                slice_indices=slice_indices,
            )

            loss, rec_error = torch.tensor(0.0), torch.tensor(0.0)  # IGNORE LOSS CALCULATION FOR 3D INFERENCE
            total_loss += loss.item() + 0.01*rec_error.item()
            metric_i = np.asarray(metric_i, dtype=np.float64)
            metric_rows.append(metric_i)
            case_mean = np.nanmean(metric_i, axis=0)
            logging.info(
                'idx %02d case %s mean_dice %f mean_hd95 %f, mean_jacard %f mean_asd %f'
                % (i_batch, case_name, case_mean[0], case_mean[1], case_mean[2], case_mean[3])
            )

        metric_stack = np.stack(metric_rows, axis=0)  # [N_cases, C-1, 4]
        # Class-wise: average over cases where that organ was present in the scored slices.
        class_means = np.nanmean(metric_stack, axis=0)
        for i in range(1, args.num_classes):
            logging.info(
                'Mean class (%d) mean_dice %f mean_hd95 %f, mean_jacard %f mean_asd %f'
                % (i, class_means[i - 1][0], class_means[i - 1][1], class_means[i - 1][2], class_means[i - 1][3])
            )
        # Dataset mean = mean over organs (Synapse protocol), ignoring NaN organs.
        mean_dice = float(np.nanmean(class_means[:, 0]))
        mean_hd95 = float(np.nanmean(class_means[:, 1]))
        logging.info(
            'Validation performance (%s) mean_dice:%f mean_hd95:%f' % (mode, mean_dice, mean_hd95))
        logging.info("Validation finished.")

        total_loss = total_loss / len(loader)
        return loss, mean_dice, mean_hd95


def inference_2d(args, net, loader, criterion, validation=False, logging=None):
    net.eval()
    total_loss = 0
    total_input = 0
    total_dice, total_miou, total_acc, total_hd95, total_jc, total_asd = 0, 0, 0, 0, 0, 0
    lambda_rec_loss = getattr(args, "lambda_rec_loss", LAMBDA_REC)
    with torch.no_grad():
        for batch in loader:
            img, lbl = batch["image"].cuda(), batch["label"].cuda()
            
            if args.amp:
                with torch.autocast(device_type="cuda"):
                    outputs = net(img)
                    loss, rec_error = calc_loss(criterion, outputs, lbl)
            else:
                outputs = net(img)                
                loss, rec_error = calc_loss(criterion, outputs, lbl)
            total_loss += loss.item() + lambda_rec_loss * rec_error.item()
            # compute dice, accumulate in total_dice ...
            if isinstance(outputs, tuple):
                outputs, rec_loss = outputs[0], outputs[1]
            if isinstance(outputs, list):
                logits = outputs[-1]
            else:
                logits = outputs
                
            metrics = compute_segmentation_metrics_hard(logits, lbl, ignore_index=None, include_background=False, eps=1e-8)
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
            B = lbl.shape[0]
            labels_for_accuracy = lbl[:, 0] if lbl.ndim == 4 and lbl.size(1) == 1 else lbl
            pixel_accuracy = (
                logits.argmax(dim=1) == labels_for_accuracy.long()
            ).float().mean().item()
            total_dice += metrics["mean_dice"] * B
            total_miou += metrics["mean_iou"] * B
            total_acc += pixel_accuracy * B
            total_hd95 += metrics["mean_hd95"] * B
            total_jc += metrics["mean_jaccard"] * B
            total_asd += metrics["mean_asd"] * B
            
            total_input += B

            
    if validation: net.train()
    avg_loss = total_loss / max(total_input, 1)
    avg_dcs = total_dice / max(total_input, 1)
    avg_miou = total_miou / max(total_input, 1)
    avg_acc = total_acc / max(total_input, 1)
    avg_hd95 = total_hd95 / max(total_input, 1)
    avg_jc = total_jc / max(total_input, 1)
    avg_asd = total_asd / max(total_input, 1)
    return avg_loss, avg_dcs, avg_miou, avg_acc, avg_hd95, avg_jc, avg_asd


def calc_loss(criterion, outputs, label_batch):
    rec_error = torch.tensor(0.0, device=label_batch.device)
    if isinstance(outputs, tuple):
        outputs, rec_error = outputs[0], outputs[1]
        if isinstance(rec_error, torch.Tensor) and rec_error.ndim > 0:
            rec_error = rec_error.mean()
    if isinstance(outputs, list):
        outputs = outputs[-1]
    loss = criterion(outputs, label_batch[:])
    return loss, rec_error


def train(args, net, loaders, optimizer, scheduler, criterion, logging, snapshot_path, writer=None, scaler=None, resume_state=None):

    print("AMP enabled...") if args.amp else print("AMP disabled...")
    if args.amp and scaler is None:
        scaler = GradScaler()

    tr_loader, vl_loader, te_loader = loaders['train'], loaders.get('val', None), loaders.get('test', None)

    global_iter_num = 0
    te_accuracy = []
    best_dcs_vl = 0.0
    best_dcs_te = 0.0
    best_ep_idx = -1
    best_dcs_fast = 0.0
    best_ep_fast = -1
    start_epoch = 0
    if resume_state is not None:
        start_epoch = int(resume_state.get("start_epoch", 0))
        global_iter_num = int(resume_state.get("global_iter_num", 0))
        best_dcs_vl = float(resume_state.get("best_dcs_vl", 0.0))
        best_dcs_te = float(resume_state.get("best_dcs_te", 0.0))
        best_ep_idx = int(resume_state.get("best_ep_idx", -1))
        best_dcs_fast = float(resume_state.get("best_dcs_fast", 0.0))
        best_ep_fast = int(resume_state.get("best_ep_fast", -1))
        _is_synapse = args.dataset_name.lower() == "synapse"
        _best_name = "best val Dice"
        logging.info(
            f"Continuing from epoch {start_epoch + 1} "
            f"(completed {start_epoch}/{args.max_epochs}), {_best_name}={best_dcs_te:.4f}"
            + (f", best fast Dice={best_dcs_fast:.4f}" if _is_synapse else "")
        )
    lambda_rec_loss = getattr(args, "lambda_rec_loss", LAMBDA_REC)

    def checkpoint_state_dict():
        model_ref = net.module if hasattr(net, "module") else net
        return model_ref.state_dict()

    def _save_last_checkpoint(epoch_completed):
        ckpt = {
            "epoch": int(epoch_completed),
            "model": checkpoint_state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "best_dcs_vl": float(best_dcs_vl),
            "best_dcs_te": float(best_dcs_te),
            "best_ep_idx": int(best_ep_idx),
            "best_dcs_fast": float(best_dcs_fast),
            "best_ep_fast": int(best_ep_fast),
            "global_iter_num": int(global_iter_num),
        }
        last_path = os.path.join(snapshot_path, "last.pth")
        torch.save(ckpt, last_path)
        logging.info(f"save resume checkpoint to {last_path} (epoch={epoch_completed})")

    for epoch in range(start_epoch, args.max_epochs):
        net.train()
        epoch_idx = epoch + 1
        train_loss = 0
        total_rec_error = 0
        lr_ = 0; lr_enc = 0;
        for i_batch, sampled_batch in enumerate(tr_loader):
            image_batch, label_batch = sampled_batch["image"], sampled_batch["label"]
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()

            if writer is not None:
                writer.global_step = global_iter_num

            optimizer.zero_grad(set_to_none=True)
            grad_norm = torch.tensor(0.0)
            if args.amp:
                with autocast(device_type='cuda'):
                    outputs = net(image_batch)
                    criterion_loss, rec_error = calc_loss(criterion, outputs, label_batch)
                    total_loss = criterion_loss + lambda_rec_loss*rec_error
                if not torch.isfinite(total_loss.detach()).all():
                    message = (
                        f"Non-finite training loss at epoch {epoch_idx}, batch {i_batch}, "
                        f"iteration {global_iter_num}: total={total_loss.detach().item()}, "
                        f"criterion={criterion_loss.detach().item()}, rec={rec_error.detach().item()}"
                    )
                    logging.error(message)
                    raise FloatingPointError(message)
                scaler.scale(total_loss).backward()
                # Must unscale before clipping; otherwise clip sees loss-scaled grads
                # and the effective update is reduced by ~loss_scale.
                scaler.unscale_(optimizer)
                max_grad_norm = getattr(args, "max_grad_norm", 0.5)
                if max_grad_norm and max_grad_norm > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        net.parameters(), max_norm=max_grad_norm
                    )
                # scaler.step skips the optimizer step on inf/NaN grads.
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = net(image_batch)
                criterion_loss, rec_error = calc_loss(criterion, outputs, label_batch)
                total_loss = criterion_loss + lambda_rec_loss*rec_error
                if not torch.isfinite(total_loss.detach()).all():
                    message = (
                        f"Non-finite training loss at epoch {epoch_idx}, batch {i_batch}, "
                        f"iteration {global_iter_num}: total={total_loss.detach().item()}, "
                        f"criterion={criterion_loss.detach().item()}, rec={rec_error.detach().item()}"
                    )
                    logging.error(message)
                    raise FloatingPointError(message)
                total_loss.backward()
                max_grad_norm = getattr(args, "max_grad_norm", 0.5)
                if max_grad_norm and max_grad_norm > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        net.parameters(), max_norm=max_grad_norm
                    )
                if not torch.isfinite(grad_norm).all():
                    message = (
                        f"Non-finite gradient norm at epoch {epoch_idx}, batch {i_batch}, "
                        f"iteration {global_iter_num}: grad_norm={grad_norm.detach().item()}"
                    )
                    logging.error(message)
                    raise FloatingPointError(message)
                optimizer.step()

            total_rec_error += rec_error.item()
            train_loss += total_loss.item()

            _lrs = scheduler.get_last_lr()
            if len(_lrs) == 2: lr_enc, lr_ = _lrs[0], _lrs[1]
            else: lr_ = _lrs[0]; lr_enc = lr_
            if writer is not None:
                writer.add_scalar('info/lr', lr_, global_iter_num)
                writer.add_scalar('info/lr_enc', lr_enc, global_iter_num)
                writer.add_scalar('info/criterion_loss', criterion_loss, global_iter_num)
                writer.add_scalar('info/rec_error', rec_error, global_iter_num)
                writer.add_scalar('info/total_loss', total_loss, global_iter_num)
                writer.add_scalar('info/grad_norm', grad_norm, global_iter_num)

            scheduler.step()

            global_iter_num = global_iter_num + 1
            if global_iter_num % 20 == 0:
                logging.info('iteration %d : loss : %f lr: %f lr_enc: %f' % (global_iter_num, total_loss.item(), lr_, lr_enc))

        tr_loss = train_loss / len(tr_loader)
        tr_dcs = tr_miou = tr_acc = tr_hd95 = tr_jc = tr_asd = 0

        # VALIDATION (skin / BUSI): used for checkpoint selection — never peek at test.
        # SYNAPSE: no separate val split; test volumes are monitored below.
        is_synapse = args.dataset_name.lower() == "synapse"
        use_val_for_best = (not is_synapse) and (vl_loader is not None)

        vl_loss = vl_dcs = vl_miou = vl_acc = vl_hd95 = vl_jc = vl_asd = 0
        if vl_loader is not None and (epoch_idx % args.val_interval == 0):
            vl_loss, vl_dcs, vl_miou, vl_acc, vl_hd95, vl_jc, vl_asd = inference_2d(
                args, net, vl_loader, criterion, validation=True
            )
            logging.info(
                f"Validation at epoch {epoch_idx}: loss={vl_loss:.4f}, dcs={vl_dcs:.4f}, "
                f"miou={vl_miou:.4f}, acc={vl_acc:.4f}, hd95={vl_hd95:.4f}, jc={vl_jc:.4f}, asd={vl_asd:.4f}"
            )
            if writer is not None:
                writer.add_scalar('val-metrics/dice', vl_dcs, epoch_idx)
                writer.add_scalar('val-metrics/miou', vl_miou, epoch_idx)
                writer.add_scalar('val-metrics/acc', vl_acc, epoch_idx)
                writer.add_scalar('val-metrics/hd95', vl_hd95, epoch_idx)
                writer.add_scalar('val-metrics/jc', vl_jc, epoch_idx)
                writer.add_scalar('val-metrics/asd', vl_asd, epoch_idx)

            if use_val_for_best and vl_dcs >= best_dcs_vl:
                best_net = copy.copy(net)
                best_dcs_vl = vl_dcs
                best_dcs_te = vl_dcs  # track "best selection score" for logging/rename
                best_ep_idx = epoch_idx
                save_model_path = os.path.join(snapshot_path, 'best.pth')
                torch.save(checkpoint_state_dict(), save_model_path)
                logging.info(
                    f"save best.pth (val Dice={100*vl_dcs:.3f}) to {save_model_path}"
                )

        # SYNAPSE: the 12 volumes are 3D validation (no held-out test).
        # Fast validation selects best.pth. The last epoch scores that set in full
        # and does not replace the selected checkpoint.
        te_loss = te_dcs = te_miou = te_acc = te_hd95 = te_jc = te_asd = 0
        scheduled_val = (
            epoch_idx >= _SYNAPSE_VAL_AFTER and epoch_idx % _SYNAPSE_VAL_INTERVAL == 0
        )
        run_synapse_val = (
            is_synapse
            and te_loader is not None
            and (scheduled_val or epoch_idx == args.max_epochs)
        )
        if run_synapse_val:
            want_fast = epoch_idx < args.max_epochs
            eval_mode = "fast" if want_fast else "full"
            te_loss, te_dcs, te_hd95 = inference_3d(
                args,
                net,
                te_loader,
                test_save_path=None,
                epoch=epoch_idx,
                logging=logging,
                fast_val=want_fast,
            )
            vl_tag = "fast validation" if eval_mode == "fast" else "full validation"
            te_res_info = f"vl_DCS:{100*te_dcs:.3f}, vl_HD95:{te_hd95:.4f} [{vl_tag}]"
            print(f"epoch:{epoch_idx:03d}/{args.max_epochs}, {te_res_info}")
            logging.info(f"Synapse 3D val at epoch {epoch_idx}: {te_res_info}")
            if writer is not None:
                writer.add_scalar('val-metrics/dice', te_dcs, epoch_idx)
                writer.add_scalar('val-metrics/hd95', te_hd95, epoch_idx)

            # best.pth is the best fast-validation checkpoint. The last-epoch
            # full validation is only a report; it does not replace best.pth.
            if eval_mode == "fast" and te_dcs >= best_dcs_fast:
                best_dcs_fast = te_dcs
                best_ep_fast = epoch_idx
                best_dcs_te = te_dcs
                best_ep_idx = epoch_idx
                best_net = copy.copy(net)
                torch.save(checkpoint_state_dict(), os.path.join(snapshot_path, "best.pth"))
                logging.info(
                    f"save best.pth from fast validation at epoch {epoch_idx}: "
                    f"Dice={100*te_dcs:.3f} HD95={te_hd95:.4f}"
                )
            te_accuracy.append(te_dcs)

        
        if writer is not None:
            writer.add_scalar('loss/train', tr_loss, epoch_idx)
            if not is_synapse:
                writer.add_scalar('loss/val', vl_loss, epoch_idx)
                writer.add_scalars('loss/all', {'train': tr_loss, 'val': vl_loss}, epoch_idx)
                writer.add_scalars('metrics-all/dice', {'train': tr_dcs, 'val': vl_dcs}, epoch_idx)
                writer.add_scalars('metrics-all/accuracy', {'train': tr_acc, 'val': vl_acc}, epoch_idx)
                writer.add_scalars('metrics-all/miou', {'train': tr_miou, 'val': vl_miou}, epoch_idx)
                writer.add_scalars('metrics-all/hd95', {'train': tr_hd95, 'val': vl_hd95}, epoch_idx)
                writer.add_scalars('metrics-all/jc', {'train': tr_jc, 'val': vl_jc}, epoch_idx)
                writer.add_scalars('metrics-all/asd', {'train': tr_asd, 'val': vl_asd}, epoch_idx)
            elif run_synapse_val:
                writer.add_scalar('loss/val', te_loss, epoch_idx)
                writer.add_scalars('loss/all', {'train': tr_loss, 'val': te_loss}, epoch_idx)
                writer.add_scalars('metrics-all/dice', {'train': tr_dcs, 'val': te_dcs}, epoch_idx)
                writer.add_scalars('metrics-all/hd95', {'train': tr_hd95, 'val': te_hd95}, epoch_idx)

        logging.info(f"Epoch {epoch_idx} training complete. Average Loss: {tr_loss:.4f}, LR: {lr_:0.9f} (LRenc: {lr_enc:0.9f})")
        log_info = [f"epoch:{epoch_idx:03d}/{args.max_epochs}",
                    f"tr-loss:{tr_loss:0.5f}",
                    f"lr:{lr_:0.6f}",
                    f"lr_enc:{lr_enc:0.7f}"]
        if not is_synapse:
            log_info += [
                f"vl-loss:{vl_loss:0.5f}",
                f"vl_DCS:{vl_dcs*100:0.3f}",
                f"vl_ACC:{vl_acc*100:0.3f}",
                f"vl_mIoU:{vl_miou*100:0.3f}",
                f"vl_HD95:{vl_hd95:0.3f}",
                f"vl_JC:{vl_jc:0.3f}",
                f"vl_ASD:{vl_asd:0.3f}",
                f"best-vl-DCS:{best_dcs_vl*100:0.3f}",
            ]
        elif run_synapse_val:
            log_info += [
                f"vl_DCS:{te_dcs*100:0.3f}",
                f"vl_HD95:{te_hd95:0.3f}",
                f"best-vl-DCS:{best_dcs_fast*100:0.3f}",
            ]
        elif is_synapse:
            log_info += [f"best-vl-DCS:{best_dcs_fast*100:0.3f}"]
        print(', '.join(log_info))

        # Always keep a resumable checkpoint (crash / migrate to another machine).
        _save_last_checkpoint(epoch_idx)

        if epoch_idx == args.max_epochs: # save the model at the last epoch
            # rename best model at the last epoch to include best dcs info
            if 'best_net' in locals() and os.path.exists(os.path.join(snapshot_path, 'best.pth')):
                _best_score = best_dcs_fast if is_synapse else best_dcs_te
                _best_ep = best_ep_fast if is_synapse else best_ep_idx
                best_model_path = os.path.join(
                    snapshot_path,
                    f'best_dcs={_best_score}_mxep={_best_ep}_lr={args.lr}_enlr={args.lr_enc}.pth',
                )
                os.rename(os.path.join(snapshot_path, 'best.pth'), best_model_path)
                logging.info("rename best model to {}".format(best_model_path))
                os.symlink(best_model_path, os.path.join(snapshot_path, 'best.pth'))

            # save last model
            _last_dcs = te_dcs if is_synapse else vl_dcs
            save_model_path = os.path.join(snapshot_path, f'last_{epoch_idx}_dcs={_last_dcs:.4f}.pth')
            torch.save(checkpoint_state_dict(), save_model_path)
            logging.info("save last model to {}".format(save_model_path))

    return {"best_epoch": best_ep_idx, "best_dsc": best_dcs_te}
