# load a well calibrated feature extractor g
# training loader D: X x y
# randomly load a z~X
# eps =1e-6, min(||g(z+eps), g(x)||)

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torch.nn.functional as F
import Data.cifar10 as cifar10
from Net.resnet import resnet50 
from Net.resnet2 import ResNet18
import matplotlib.pyplot as plt
import torch.backends.cudnn as cudnn
import numpy as np
import time

# set seed
torch.manual_seed(1)
np.random.seed(1)
cudnn.deterministic = True

# hyperparameters
batch_size = 1
use_aug = False
n_iters = 1000
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# cifar10 label to class array
cifar10_labels = np.array([
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck',
])
pretrained_weights = {"ce":"weights/cifar10/resnet50_cross_entropy_350.model",
                        "fl":"weights/cifar10/resnet50_focal_loss_gamma_3.0_350.model",
                        "ls":"weights/cifar10/resnet50_cross_entropy_smoothed_smoothing_0.05_350.model",
                        "resnet18_pgd":"weights/cifar10/resnet18_pgd_adversarial_training",
                        "resnet18_basic":"weights/cifar10/basic_training",}
key = "fl"
# generate z_calibrated
def calibrate_feature(model, z, x_feature, eps=0.5, alpha=0.5/5, iters=7):
    z = z.to(device)
    x_feature = x_feature.detach().to(device)
    for i in range(n_iters):
        z.requires_grad = True
        logit, z_feature = model(z)
        model.zero_grad()
        z.grad = None

        cost = torch.norm(z_feature - x_feature)
        cost.backward()
        
        
        norm_grad = z.grad / (torch.norm(z.grad) + 1e-10)
        adv_z = z - alpha*norm_grad
        
        # adv_z = z - alpha*z.grad.sign()

        z = torch.clamp(adv_z, min=0, max=1).detach_()
        if i % 100 == 0:
            print(f'Iter {i}, cost: {cost.item()}')
    return z


def single_pgd_step_robust(model, rand_sample, robust_feature, alpha, delta):
    delta.requires_grad = True
    logit, feature = model(rand_sample + delta)
    loss = torch.norm((feature - robust_feature))
    
    loss.mean().backward(retain_graph=True)
    grad = delta.grad.data

    norm_grad = torch.norm(grad)
    
    delta = delta - alpha * grad / (norm_grad + 1e-10)
    
    delta = torch.clamp(delta, -rand_sample, 1 - rand_sample)
    
    delta = delta.detach()
    
    return delta, loss

def pgd_l2_robust(model, rand_sample, robust_feature, alpha, num_iter, epsilon=0, example=False):
    delta = torch.zeros_like(rand_sample)
    loss = 0
    for i in range(num_iter):
        delta, loss = single_pgd_step_robust(model, rand_sample, robust_feature, alpha, delta)
        if i % 100 == 0:
            print(f'Iter {i}, cost: {loss.item()}')
    
    if example:
        # Printing the average loss across the batch for clarity
        print(f'{num_iter} iterations, final MSE {loss.mean().item()}')
    return delta


# visualize as a cifar10 imgae
def visualize(x, title, name):
    x = x.squeeze(0)
    x = x.cpu().detach().numpy()
    x = x.transpose(1,2,0)
    # unnormalize     
    plt.title(name + "----" +title)
    plt.imshow(x)
    plt.savefig(name)
    plt.close()

# load original cifar10 
train_loader, val_loader = cifar10.get_train_valid_loader(
            batch_size=batch_size,
            augment=use_aug,
            random_seed=1,
        )

test_loader = cifar10.get_test_loader(
    batch_size=batch_size,
)


# load a well calibrated feature extractor g
# for focal calibration pretrained weight
model = resnet50(num_classes=10)
weight = torch.load(pretrained_weights[key]) 

# for pgd adversarial training resnet18 pretrained weight
# model = ResNet18()
# weight = torch.load(pretrained_weights[key])['net'] 
# load one device model with DataParallel weights
weight = {k.replace('module.', ''): v for k, v in weight.items()}
model.load_state_dict(weight)
model.to(device)

for (x,y) in train_loader:
    # randomly select a z from the training set
    random_n = torch.randint(0, len(train_loader.dataset), (1,))

    z, label_z = train_loader.dataset[random_n]
    
    z = z.unsqueeze(0).to(device)
    x = x.to(device)
    logitm, x_feature = model(x)
    z_calibrated = calibrate_feature(model, z, x_feature, n_iters)
    # learned_delta = pgd_l2_robust(model, z, x_feature, alpha=0.1, num_iter=n_iters)
    # z_calibrated = z + learned_delta

    visualize(x, cifar10_labels[y.item()], "output/x.png")
    visualize(z, cifar10_labels[label_z], "output/z.png")
    visualize(z_calibrated, cifar10_labels[y.item()],"output/calibrated_z.png")

    print("generated")