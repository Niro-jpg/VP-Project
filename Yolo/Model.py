import torch
from config import *
import numpy as np

#funzione che calcola la famosa loss di yolo 
# pred ed y hanno dimensione (n,S,S,(B*5+C))
def YOLO_Loss(pred, y):
    B = int(pred.shape[3]/5)
    pred = pred.view(pred.shape[0]*pred.shape[1]*pred.shape[2], pred.shape[3])
    y = y.view(y.shape[0]*y.shape[1]*y.shape[2],y.shape[3])

    obj_indexis = [index for index, tensor in enumerate(y) if tensor[0] == 1]
    obj_y = y[obj_indexis]
    obj_pred = pred[obj_indexis]
    #obj_y = y
    #obj_pred = pred
    obj_n = len(obj_indexis)
    #obj_n =pred.shape[0]

    noobj_indexis = [index for index, tensor in enumerate(y) if tensor[0] == 0]
    noobj_y = y[noobj_indexis]
    noobj_pred = pred[noobj_indexis]
    #noobj_y = y
    #noobj_pred = pred
    noobj_n = len(noobj_indexis)

    IOUS = ([IOU(obj_pred[:,1 + i*5],obj_y[:,1],obj_pred[:,2 + i*5],obj_y[:,2],obj_pred[:,3],obj_y[:,3],obj_pred[:,4],obj_y[:,4]) for i in range(B)])
    IOUS_argmax = torch.stack([tensor for tensor in IOUS]).argmax(dim = 0)
    obj_pred = torch.cat([torch.cat((obj_pred[i, IOUS_argmax[i]*5:(IOUS_argmax[i]*5)+5],obj_pred[i,B*5:B*5+3]))for i in range(IOUS_argmax.shape[0])]).view(obj_n, 8)

    center_loss = ((obj_pred[:,1] - obj_y[:,1])**2 + (obj_pred[:,2] - obj_y[:,2])**2).sum()
    dim_loss = ((obj_pred[:,3].abs().sqrt()*torch.sign(obj_pred[:,3]) - obj_y[:,3].sqrt())**2 + (obj_pred[:,4].abs().sqrt()*torch.sign(obj_pred[:,4]) - obj_y[:,4].sqrt())**2).sum()
    confidence_loss = ((obj_pred[:,0] - obj_y[:,0])**2).sum()
    no_confidence_loss_tensor = torch.stack([torch.stack(([tensor[i*5] for i in range(B)])).max() for tensor in noobj_pred])
    no_confidence_loss = ((no_confidence_loss_tensor)**2).sum()
    #probability_loss = (((obj_pred[:,5:8] - obj_y[:,5:8])**2).sum(dim = 1)*obj_y[:,0]).mean()
    class_loss = ((obj_pred[:,-3:] - obj_y[:,-3:])**2).sum()
    probability_loss = 0
    loss_sum = 5*center_loss + confidence_loss + 0.5*no_confidence_loss + probability_loss + 5*dim_loss + class_loss
    return loss_sum

def IOU(x1,x2,y1,y2,w1,w2,h1,h2):
    xinf1 = x1 - w1/2
    xinf2 = x2 - w2/2
    xsup1 = x1 + w1/2
    xsup2 = x2 + w2/2
    yinf1 = y1 -h1/2
    yinf2 = y2 - h2/2
    ysup1 = y1 + h1/2
    ysup2 = y2 + h2/2
    int_xinf = torch.max(xinf1,xinf2)
    int_xsup = torch.min(xsup1,xsup2)
    int_yinf = torch.max(yinf1,yinf2)
    int_ysup = torch.min(ysup1,ysup2)
    int_area = (int_xsup-int_xinf)*(int_ysup-int_yinf)
    un_area = ((w1*h1)+(w2*h2))-int_area
    iou = int_area/un_area
    return iou

def suppression(tensors, n, S, B):
    print(tensors)
    mask = n%5 == 0
    con_list = tensors[mask]
    print(con_list)