# -*- coding: utf-8 -*-
# 作者：韩信
# github地址：https://github.com/hanxinif

import torch
import torch.nn as nn

class focaleiou_iouloss(nn.Module):
    """
    Calculate Intersection over Union (IoU) of box1(1, 4) to box2(n, 4).

    Args:
        box1 (torch.Tensor): A tensor representing a single bounding box with shape (1, 4).
        box2 (torch.Tensor): A tensor representing n bounding boxes with shape (n, 4).
        xywh (bool, optional): If True, input boxes are in (x, y, w, h) format. If False, input boxes are in
                               (x1, y1, x2, y2) format. Defaults to True.
        GIoU (bool, optional): If True, calculate Generalized IoU. Defaults to False.
        DIoU (bool, optional): If True, calculate Distance IoU. Defaults to False.
        CIoU (bool, optional): If True, calculate Complete IoU. Defaults to False.
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-7.

    Returns:
        (torch.Tensor): IoU, GIoU, DIoU, or CIoU values depending on the specified flags.
    """

    def __init__(self, reduction='none'):
        super().__init__()
        self.reduction = reduction

    def forward(self, box1, box2, xywh=True, gamma=0.8, alpha=1, eps=1e-7):

        assert box1.shape[0] == box2.shape[0]

        # Get the coordinates of bounding boxes
        if xywh:  # transform from xywh to xyxy
            (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
            w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
            b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
            b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
        else:  # x1, y1, x2, y2 = box1
            b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
            b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
            w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
            w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

        #################

        pred = box1.view(-1, 4)
        target = box2.view(-1, 4)
        tl = torch.max(
            (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
        )
        br = torch.min(
            (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
        )

        area_p = torch.prod(pred[:, 2:], 1)
        area_g = torch.prod(target[:, 2:], 1)

        en = (tl < br).type(tl.type()).prod(dim=1)
        area_i = torch.prod(br - tl, 1) * en
        area_u = area_p + area_g - area_i
        iou = ((area_i) / (area_u + 1e-16)).view(-1, 1)

        ###################

        cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)  # convex (smallest enclosing box) width
        ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)  # convex height

        c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
        rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center dist ** 2

        rho_w2 = ((b2_x2 - b2_x1) - (b1_x2 - b1_x1)) ** 2
        rho_h2 = ((b2_y2 - b2_y1) - (b1_y2 - b1_y1)) ** 2
        cw2 = torch.pow(cw ** 2 + eps, alpha)
        ch2 = torch.pow(ch ** 2 + eps, alpha)

        focaleiou = iou - (rho2 / c2 + rho_w2 / cw2 + rho_h2 / ch2), torch.pow(iou, gamma)  # Focal_EIou
        loss = (1.0 - focaleiou[0]) * focaleiou[1].detach()

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss

if __name__ == '__main__':

    pred_boxes = torch.tensor([[25, 25, 75, 75], [50, 50, 100, 100]], dtype=torch.float32)
    target_boxes = torch.tensor([[30, 30, 70, 70], [70, 70, 120, 120]], dtype=torch.float32)
    block = focaleiou_iouloss(reduction='none')
    loss = block(pred_boxes, target_boxes)
    print(f"Focal_EIoU Loss: {loss}")