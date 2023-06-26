import torch
import warnings
import torchvision
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.models import vgg19
warnings.filterwarnings("ignore")

class Neural_Style(nn.Module):
  
  def __init__(self):
    super(Neural_Style, self).__init__()
    self.nnet = vgg19(pretrained = True).features

  def forward(self,x):
    feature_space = []
    for i,layer in enumerate(self.nnet):
      x = layer(x)
      if i in [4,9,18,27,36]:
        feature_space.append(x)
    return feature_space

def style(original,style):

    transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((300,400)),
    torchvision.transforms.ToTensor()
    ])

    original = transform(original)
    style = transform(style)
    generate = original.clone().requires_grad_(True)

    model = Neural_Style()
    epochs = 20
    alpha = 1
    beta = 1e3
    optimizer = torch.optim.Adam([generate], lr = 1e-2)

    for epoch in range(epochs):

        optimizer.zero_grad()
        generate_features = model(generate)
        original_features = model(original)
        style_features = model(style)
        content_loss = 0
        style_loss = 0
        for i in range(5):
            content_loss += torch.mean((generate_features[i] - original_features[i])**2)
            style_loss += torch.mean((torch.matmul(generate_features[i].view(generate_features[i].shape[0],-1),
                                                generate_features[i].view(generate_features[i].shape[0],-1).T)
                                                /(generate_features[i].shape[0]*generate_features[i].shape[1])
                                - torch.matmul(style_features[i].view(style_features[i].shape[0],-1),
                                            style_features[i].view(style_features[i].shape[0],-1).T)
                                            /(style_features[i].shape[0]*style_features[i].shape[1]))**2)
        loss = alpha*content_loss + beta*style_loss
        loss.backward()
        optimizer.step()
    #print("Iteration Number - ",epoch)
    #print("Content Loss - ",content_loss)
    #print("Style Loss - ", style_loss)
    #print("Loss - ",loss)
    output_transform = torchvision.transforms.ToPILImage()
    return output_transform(generate)