import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.autograd import Variable

import torchvision
from torchvision import transforms as T
from torchvision import models

import cv2
import numpy as np

import argparse
import os
import sys

import model_utils as mu

def gram_matrix(features):
    n, c, h, w = features.shape
    features = features.view(c, h*w)
            
    return features.matmul(features.T)

def style_loss(img_features, style_features):  
    loss = 0
    
    for img_feature, style_feature in zip(img_features, style_features):
        img_gram = gram_matrix(img_feature)
        style_gram = gram_matrix(style_feature)

        loss += F.l1_loss(img_gram, style_gram)
        
    return loss

def main(args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    architectures = {
        'vgg16': models.vgg16,
        'squeezenet': models.squeezenet1_0,
    }
    feature_map_fns = {
        'vgg16': mu.vgg16_feature_maps,
        'squeezenet': mu.squeezenet_feature_maps,
    }
    model = architectures[args.model](pretrained=True).to(device)
    feature_map_fn = feature_map_fns[args.model]
    
    # no need to compute gradients for vgg16
    for param in model.parameters():
        param.requires_grad = False
        
    # initial reading of input images
    target_content = cv2.imread(args.content_loc, cv2.IMREAD_COLOR)
    target_style = cv2.imread(args.style_loc, cv2.IMREAD_COLOR)
    
    if args.dimensions is not None:
        shape = tuple(args.dimensions)[::-1]
    else:
        # grab only height and width
        # reverse to width and height for resizing in opencv
        shape = target_content.shape[:2][::-1]

    # convert to RGB and resize content image
    target_content = cv2.cvtColor(cv2.resize(target_content, shape), cv2.COLOR_BGR2RGB)
    target_style = cv2.cvtColor(target_style, cv2.COLOR_BGR2RGB)
    
    if args.tile:
        H, W, C = target_content.shape
        h, w, c = target_style.shape

        y = H/h
        y = int(y) if y == int(y) else int(y+1)

        x = W/w
        x = int(x) if x == int(x) else int(x+1)
        
        target_style = np.tile(target_style, (y, x, 1))[:H, :W]
    else:
        target_style = cv2.resize(target_style, (target_content.shape[:2][::-1]))
        
    # tensor shape
    shape = (1, 3, target_content.shape[0], target_content.shape[1])

    # convert input images to tensors and load onto device
    target_content_ = T.ToTensor()(target_content).view(shape).to(device)
    target_style_ = T.ToTensor()(target_style).view(shape).to(device)

    # initialize output image to content image
    img = Variable(target_content_, requires_grad=True)
    if args.learning_rate:
        optimizer = optim.Adam([img], args.learning_rate)
    else:
        optimizer = optim.Adam([img])
        
    #print(np.array(T.ToPILImage()(img[0].detach().cpu())))
    content_features = feature_map_fn(model, target_content_)
    style_features = feature_map_fn(model, target_style_)
    
    for epoch in range(args.epochs):
        img.grad = None
        features = feature_map_fn(model, img)

        content_loss_ = F.l1_loss(features[-1], content_features[-1])
        style_loss_ = style_loss(features, style_features)
        total_loss = args.alpha * content_loss_ + args.beta * style_loss_
        
        total_loss.backward()
        optimizer.step()

        # keep values on the interval [0, 1] to correctly convert to rgb image
        img.data.clamp_(0, 1)
        
        sys.stdout.write(f'{epoch+1}) C Loss: {content_loss_:.4f} | S Loss: {style_loss_:.4f}                             \r')
        sys.stdout.flush()
        
        if args.interval is not None and epoch > 0 and epoch % args.interval == 0:
            out_path = os.path.join(args.output_dir, f'{args.name}_{epoch}.png')
            print(f'writing to {out_path}')
            cv2.imwrite(out_path, np.array(T.ToPILImage()(img[0].detach().cpu()))[..., ::-1])
            
    out_path = os.path.join(args.output_dir, f'{args.name}.png')
    print(f'writing to {out_path}')
    cv2.imwrite(out_path, np.array(T.ToPILImage()(img[0].detach().cpu()))[..., ::-1])
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--content-loc', required=True, help='path to content image')
    parser.add_argument('-s', '--style-loc', required=True, help='path to style image')
    parser.add_argument('-d', '--dimensions', nargs='+', type=int,
                        help='''height and width of output image in pixels. 
                                defaults to dimensions of content image if none is given.''')
    parser.add_argument('-o', '--output-dir', default='.', help='location to save output to')
    parser.add_argument('-i', '--interval', type=int,
                        help='''saves output to output directory every N epochs
                                where N is the value given''')
    parser.add_argument('-a', '--alpha', type=float, default=1, help='content weighting factor')
    parser.add_argument('-b', '--beta', type=float, default=1000000, help='style weighting factor')
    parser.add_argument('-t', '--tile', action='store_true', help='tile style image instead of resizing it')
    parser.add_argument('-e', '--epochs', default=1000, type=int, help='number of optimization steps')
    parser.add_argument('-n', '--name', default='out', help='name of output file')
    parser.add_argument('-l', '--learning-rate', type=float, help='learning rate of model')
    parser.add_argument('-m', '--model', default='vgg16', help='`vgg16` or `squeezenet`')
    
    args = parser.parse_args()
    main(args)
    