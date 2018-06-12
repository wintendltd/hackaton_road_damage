import torch

from utils.ssd import ssd

vgg = torch.load('../models/vgg16.pth')

ssd = ssd()
layer_indices = [0, 2, 5, 7, 10, 12, 14, 17, 19, 21]

for layer_idx in layer_indices:
    ssd.base[layer_idx].weight.data = vgg['features.%d.weight' % layer_idx]
    ssd.base[layer_idx].bias.data = vgg['features.%d.bias' % layer_idx]

ssd.conv5_1.weight.data = vgg['features.24.weight']
ssd.conv5_1.bias.data = vgg['features.24.bias']
ssd.conv5_2.weight.data = vgg['features.26.weight']
ssd.conv5_2.bias.data = vgg['features.26.bias']
ssd.conv5_3.weight.data = vgg['features.28.weight']
ssd.conv5_3.bias.data = vgg['features.28.bias']

torch.save(ssd.state_dict(), 'ssd.pth')
