import torch
from config import *
import numpy as np

#funzione che calcola la famosa loss di yolo 
# pred ed y hanno dimensione (n,S,S,(B*5+C))
def YOLO_Loss(pred, y):
    B = int(pred.shape[3]/5)
    pred = pred.view(pred.shape[0]*pred.shape[1]*pred.shape[2], pred.shape[3])
    y = y.view(y.shape[0]*y.shape[1]*y.shape[2],y.shape[3])
    n = pred.shape[0]
    IOUS = ([IOU(pred[:,1 + i*5],y[:,1],pred[:,2 + i*5],y[:,2],pred[:,3],y[:,3],pred[:,4],y[:,4]) for i in range(B)])
    IOUS_argmax = torch.stack([tensor for tensor in IOUS]).argmax(dim = 0)
    pred = torch.cat([torch.cat((pred[i, IOUS_argmax[i]*5:(IOUS_argmax[i]*5)+5],pred[i,B*5:B*5+3]))for i in range(IOUS_argmax.shape[0])]).view(n,8)
    center_loss = ((pred[:,1] - y[:,1])**2 + (pred[:,2] + y[:,2])**2).sum()
    dim_loss = ((pred[:,3] - y[:,3])**2 + (pred[:,4] - y[:,4])**2).sum()
    confidence_loss = (((pred[:,0] - y[:,0])**2)*y[:,0]).sum()
    no_confidence_loss = (((pred[:,0] - y[:,0])**2)*(1 - y[:,0])).sum()
    probability_loss = (((pred[:,5:8] - y[:,5:8])**2).sum(dim = 1)*y[:,0]).sum()
    print("-----------------------------------------------------")
    loss_sum = 5*(center_loss) + confidence_loss + 0.5*no_confidence_loss + probability_loss
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