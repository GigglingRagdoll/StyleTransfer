import torch
import torch.nn as nn
import torchvision

def squeezenet_feature_maps(squeezenet, x):
    feature_maps = []
    
    pooling_layers = [2, 6, 11]
    avg_pool = nn.AvgPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)
 
    # first feature map
    x = squeezenet.features[0](x)
    feature_maps.append(x)

    # skip over first convolution since feature map is already computed
    for i in range(1, len(squeezenet.features)):
        feature = squeezenet.features[i]

        # use avg pooling instead of max pooling
        if i in pooling_layers:
            x = avg_pool(x)
            
        # handle feature maps in fire blocks
        elif type(feature) == torchvision.models.squeezenet.Fire:
            x0 = feature.squeeze(x)
            x = feature.squeeze_activation(x0)
            
            x1x1 = feature.expand1x1(x)
            x3x3 = feature.expand3x3(x)
            
            feature_maps = feature_maps + [x0, torch.cat([x1x1, x3x3], 1)]
            #feature_maps = feature_maps + [x0, x1x1, x3x3]
            
            x = torch.cat([
                feature.expand1x1_activation(x1x1), 
                feature.expand3x3_activation(x3x3)
            ], 1)
            
        # all other non feature map layers
        else:
            x = feature(x)
            
    return feature_maps

def vgg16_feature_maps(vgg16, x):
    ''' Returns feature maps of given layers from vgg16 model
    
        vgg16_model: a pretrained pytorch vgg16 model
        out_layers
    '''
    # indices of all convolutional layers
    conv_layers = [0, 2, 5, 7, 10, 12, 14, 17, 19, 21, 24, 26, 28]
    # indices of all max pooling layers
    pooling_layers = [4, 9, 16, 23, 30]
    avg_pool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=False)\
    
    feature_maps = []
    
    for i in range(len(vgg16.features)):
        feature = vgg16.features[i]
        
        if i in conv_layers:
            x = feature(x)
            feature_maps.append(x)
            
        elif i in pooling_layers:
            x = avg_pool(x)
            
        else:
            x = feature(x)
            
    return feature_maps
