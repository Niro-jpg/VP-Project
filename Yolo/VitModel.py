import torch
from Net_lite import *
from VitNet import *

# Yolo Model
class Yolo(nn.Module):
    def __init__(self, C, B=2, S=8, lambda_obj=5, lambda_noobj=0.5):
        super(Yolo, self).__init__()
        self.net = Transformer(C, B=B)
        self.B = B
        self.C = C
        self.S = S
        self.lambda_obj = lambda_obj
        self.lambda_noobj = lambda_noobj

    def forward(self, x):

        x = self.net.forward(x)
        return x

    def loss(self, pred, y):
        # preprocessing of the tensors
        pred = pred.view(pred.shape[0] * pred.shape[1] * pred.shape[2], pred.shape[3])
        y = y.view(y.shape[0] * y.shape[1] * y.shape[2], y.shape[3])
        # selecting the boxes with an image center
        obj_indexis = [index for index, tensor in enumerate(y) if tensor[0] == 1]
        obj_y = y[obj_indexis]
        obj_pred = pred[obj_indexis]
        obj_n = len(obj_indexis)
        # selecting the boxes without an image center
        noobj_indexis = [index for index, tensor in enumerate(y) if tensor[0] == 0]
        noobj_pred = pred[noobj_indexis]
        # Calculating the IOU of each region proposal and taking the max one
        IOUS = [
            self.IOU(
                obj_pred[:, 1 + i * 5],
                obj_y[:, 1],
                obj_pred[:, 2 + i * 5],
                obj_y[:, 2],
                obj_pred[:, 3],
                obj_y[:, 3],
                obj_pred[:, 4],
                obj_y[:, 4],
            )
            for i in range(self.B)
        ]
        IOUS_argmax = torch.stack([tensor for tensor in IOUS]).argmax(dim=0)
        obj_pred = torch.cat(
            [
                torch.cat(
                    (
                        obj_pred[i, IOUS_argmax[i] * 5 : (IOUS_argmax[i] * 5) + 5],
                        obj_pred[i, self.B * 5 : self.B * 5 + 3],
                    )
                )
                for i in range(IOUS_argmax.shape[0])
            ]
        ).view(obj_n, 8)
        # calculating the loss of the center of the box center proposal
        center_loss = (
            (obj_pred[:, 1] - obj_y[:, 1]) ** 2 + (obj_pred[:, 2] - obj_y[:, 2]) ** 2
        ).sum()
        # calculating the loss of the center of the box dimensions proposal
        dim_loss = (
            (
                obj_pred[:, 3].abs().sqrt() * torch.sign(obj_pred[:, 3])
                - obj_y[:, 3].sqrt()
            )
            ** 2
            + (
                obj_pred[:, 4].abs().sqrt() * torch.sign(obj_pred[:, 4])
                - obj_y[:, 4].sqrt()
            )
            ** 2
        ).sum()
        # calculating the loss of the center of the box confidence proposal
        confidence_loss = ((obj_pred[:, 0] - obj_y[:, 0]) ** 2).sum()
        # calculating the loss of the center of the box center proposal of the boxes without an object center
        no_confidence_loss_tensor = torch.stack(
            [
                torch.stack(([tensor[i * 5] for i in range(self.B)])).max()
                for tensor in noobj_pred
            ]
        )
        no_confidence_loss = ((no_confidence_loss_tensor) ** 2).sum()
        # calculating the loss of the center of the box class proposal
        class_loss = ((obj_pred[:, -3:] - obj_y[:, -3:]) ** 2).sum()
        # summing the losses with their hyperparameters
        loss_sum = (
            self.lambda_obj * center_loss
            + confidence_loss
            + self.lambda_noobj * no_confidence_loss
            + self.lambda_obj * dim_loss
            + class_loss
        )
        return loss_sum

    # Function who calculate the loss
    def IOU(self, x1, x2, y1, y2, w1, w2, h1, h2):
        xinf1 = x1 - w1 / 2
        xinf2 = x2 - w2 / 2
        xsup1 = x1 + w1 / 2
        xsup2 = x2 + w2 / 2
        yinf1 = y1 - h1 / 2
        yinf2 = y2 - h2 / 2
        ysup1 = y1 + h1 / 2
        ysup2 = y2 + h2 / 2
        int_xinf = torch.max(xinf1, xinf2)
        int_xsup = torch.min(xsup1, xsup2)
        int_yinf = torch.max(yinf1, yinf2)
        int_ysup = torch.min(ysup1, ysup2)
        int_area = (int_xsup - int_xinf) * (int_ysup - int_yinf)
        un_area = ((w1 * h1) + (w2 * h2)) - int_area
        iou = int_area / un_area
        return iou
