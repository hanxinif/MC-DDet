import torch.nn as nn
import torch

class FocalLoss(nn.Module):
    '''
    用在代替原来的BCEcls(分类损失)和BCEobj(置信度损失)
    优点:
        1.解决了单阶段目标检测中图片正负样本(前景和背景)不均衡的问题;
        2.降低简单样本的权重, 使损失函数更关注困难样本
    '''
    def __init__(self, reduction='none', gamma=1.5, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')  # 必须为nn.BCEWithLogitsLoss = sigmoid + BCELoss
        self.gamma = gamma  # 参数γ用于削弱简单样本对loss的贡献程度
        self.alpha = alpha  # 参数α用于平衡正负样本个数不均衡的问题
        self.reduction = reduction # focalloss中的BCE函数的reduction='none', 需要将focal loss应用到每个样本中

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)  # BCE(p_t) = -log(p_t)
        pred_prob = torch.sigmoid(pred)
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)  # p_t
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)  # α_t
        modulating_factor = (1.0 - p_t) ** self.gamma  # (1-p_t)^γ
        
        loss *= alpha_factor * modulating_factor  # 损失乘上系数

        # 最后选择focalloss返回的类型 默认是mean
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss

