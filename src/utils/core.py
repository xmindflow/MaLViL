
import torch
import torch.nn as nn
from torch.nn import functional as F



def get_optimizer(model, args, encoder_module_name='backbone', logging=None):
    # Separate parameter groups: encoder (pretrained) vs rest
    try:
        encoder_params = list(getattr(model, encoder_module_name).parameters())
        other_params = [p for n, p in model.named_parameters() if not n.startswith(encoder_module_name)]
    except AttributeError:
        print(f"Warning: Encoder module '{encoder_module_name}' not found. Using all parameters for optimization.")
        args.lr_enc = args.lr  # Ensure encoder LR is the same as overall LR
        encoder_params = []
        other_params = model.parameters()
    P = print if logging is None else logging.info
        
    if args.optimizer.lower() == 'adam':
        P("Using Adam optimizer")
        if args.lr_enc != args.lr:
            P(f" - Base LR: {args.lr}, Encoder LR: {args.lr_enc}")
            optimizer = torch.optim.Adam([
                {'params': encoder_params, 'lr': args.lr_enc},
                {'params': other_params, 'lr': args.lr}
            ], weight_decay=args.weight_decay)
        else:
            P(f" - Using same LR for all parameters: {args.lr}")
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        
    elif args.optimizer.lower() == 'adamw':
        P("Using AdamW optimizer")
        params = {'weight_decay': args.weight_decay, 
                  'eps': args.eps if hasattr(args, 'eps') else 1e-8,
                  'betas': (0.9, 0.999)}
        if args.lr_enc != args.lr:
            P(f" - Base LR: {args.lr}, Encoder LR: {args.lr_enc}")
            optimizer = torch.optim.AdamW([
                {'params': encoder_params, 'lr': args.lr_enc},
                {'params': other_params, 'lr': args.lr}
            ], **params)
        else:
            P(f" - Using same LR for all parameters: {args.lr}")
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, **params)
    
    elif args.optimizer.lower() == 'sgd':
        P("Using SGD optimizer with momentum 0.9")
        if args.lr_enc != args.lr:
            P(f" - Base LR: {args.lr}, Encoder LR: {args.lr_enc}")
            optimizer = torch.optim.SGD([
                {'params': encoder_params, 'lr': args.lr_enc},
                {'params': other_params, 'lr': args.lr}
            ], weight_decay=args.weight_decay, momentum=0.9)
        else:
            P(f" - Using same LR for all parameters: {args.lr}")
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.9)
    else:
        raise NotImplementedError(f"Optimizer {args.optimizer} not implemented")

    return optimizer


def get_scheduler(optimizer, args, max_iterations, logging=None):
    print(f"Using {args.scheduler} scheduler")
    if logging:
        logging.info(f"Using {args.scheduler} scheduler")

    name = args.scheduler.lower()
    if "cosine" in name:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_iterations)
    elif "poly" in name:
        power, min_lr = 0.9, 1e-6
        print(f"Polynomial LR Scheduler with power={power} and min_lr={min_lr}")
        if logging:
            logging.info(f"Polynomial LR Scheduler with power={power} and min_lr={min_lr}")
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: max((1 - step / max_iterations) ** power, min_lr),
        )
    else:
        raise NotImplementedError(f"Scheduler <{args.scheduler}> not implemented (paper: poly)")
    return scheduler

class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-6
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            # class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / sum(weight)


class BoundaryDoULoss(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes
        self.register_buffer(
            "kernel",
            torch.tensor(
                [[0,1,0],[1,1,1],[0,1,0]],
                dtype=torch.float16,
                device='cuda' if torch.cuda.is_available() else 'cpu'
            ).unsqueeze(0).unsqueeze(0)
        )

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _adaptive_size(self, score, target):
        # kernel = torch.Tensor([[0,1,0], [1,1,1], [0,1,0]])
        # padding_out = torch.zeros((target.shape[0], target.shape[-2]+2, target.shape[-1]+2))
        # padding_out[:, 1:-1, 1:-1] = target
        # h, w = 3, 3

        # Y = torch.zeros((padding_out.shape[0], padding_out.shape[1] - h + 1, padding_out.shape[2] - w + 1)).cuda()
        # for i in range(Y.shape[0]):
        #     Y[i, :, :] = torch.conv2d(target[i].unsqueeze(0).unsqueeze(0), kernel.unsqueeze(0).unsqueeze(0).cuda(), padding=1)
        kernel = self.kernel.to(target.dtype)
        Y = F.conv2d(target.unsqueeze(1), kernel, padding=1)
        Y = Y.squeeze(1)
        Y = Y * target
        Y[Y == 5] = 0
        C = torch.count_nonzero(Y)
        S = torch.count_nonzero(target)
        smooth = 1e-5
        alpha = 1 - (C + smooth) / (S + smooth)
        alpha = 2 * alpha - 1

        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        # alpha = min(alpha, 0.8)  ## We recommend using a truncated alpha of 0.8, as using truncation gives better results on some datasets and has rarely effect on others.
        alpha = torch.clamp(alpha, max=0.8)
        loss = (z_sum + y_sum - 2 * intersect + smooth) / (z_sum + y_sum - (1 + alpha) * intersect + smooth)

        return loss

    def forward(self, inputs, target):
            inputs = torch.softmax(inputs, dim=1)
            target = self._one_hot_encoder(target)

            assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(),
                                                                                                      target.size())

            loss = 0.0
            for i in range(0, self.n_classes):
                loss += self._adaptive_size(inputs[:, i], target[:, i])
            return loss / self.n_classes


class Criterion(nn.Module):
    def __init__(self, num_classes, args):
        super().__init__()
        loss_type = args.loss_type.split(',')
        loss_weights = args.loss_weights.split(',')
        
        self.lnames, self.losses, self.weights = [], [], []
        for l, w in zip(loss_type, loss_weights):
            self.weights.append(float(w))
            self.lnames.append(l)
            if l == 'dice':
                self.losses.append(DiceLoss(num_classes))
            elif l == 'boundary':
                self.losses.append(BoundaryDoULoss(num_classes))
            elif l == 'ce':
                self.losses.append(nn.CrossEntropyLoss())
            else:
                raise NotImplementedError(f"Loss {l} not implemented")    
    def forward(self, outputs, labels):
        loss = 0.0
        for w, loss_fn, l_name in zip(self.weights, self.losses, self.lnames):
            if l_name == 'ce':
                loss += w*loss_fn(outputs, labels[:].long())
            elif l_name == 'dice':
                loss += w*loss_fn(outputs, labels, softmax=True)
            elif l_name == 'boundary':
                loss += w*loss_fn(outputs, labels[:])
        return loss
