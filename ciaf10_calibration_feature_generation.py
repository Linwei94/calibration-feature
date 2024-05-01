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
import matplotlib.pyplot as plt
import torch.backends.cudnn as cudnn
import numpy as np
import time

# set seed
torch.manual_seed(1)
np.random.seed(1)
cudnn.deterministic = True

# hyperparameters
batch_size = 256
use_aug = False
n_iters = 1000
weight_name = "fl"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# pretrained_weight = "weights/cifar10/resnet50_cross_entropy_350.model"
# pretrained_weight = "weights/cifar10/resnet50_focal_loss_gamma_3.0_350.model"
# pretrained_weight = "weights/cifar10/resnet50_cross_entropy_smoothed_smoothing_0.05_350.model"
pretrained_weights = {"ce":"weights/cifar10/resnet50_cross_entropy_350.model",
                        "fl":"weights/cifar10/resnet50_focal_loss_gamma_3.0_350.model",
                        "ls":"weights/cifar10/resnet50_cross_entropy_smoothed_smoothing_0.05_350.model"}
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
        
        # pgd attack code
        # adv_z = z - alpha*z.grad.sign()


        z = torch.clamp(adv_z, min=0, max=1).detach_()
        if i % 100 == 0:
            print(f'Iter {i}, cost: {cost.item()}')
    return z

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

# cifar10 label to class array
cifar10_labels = np.array([
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck',
])



# load a well calibrated feature extractor g
model = resnet50(num_classes=10)
model.to(device)
# model = torch.nn.DataParallel(
#     model, device_ids=range(torch.cuda.device_count()))
# cudnn.benchmark = True
weight = torch.load(pretrained_weights[weight_name])
# load one device model with DataParallel weights
weight = {k.replace('module.', ''): v for k, v in weight.items()}
model.load_state_dict(weight)

        
ys = []
y_zs = []
xs = []
zs = []
zcs = []
for i, (x,y) in enumerate(train_loader):
    start_time = time.time()
    # generate 32 random numbers
    indexs = list(range(len(train_loader.dataset)))
    np.random.shuffle(indexs)
    # sample a batch of z
    z = []
    label_z = []
    batch_size = x.size(0)
    for idx in indexs[i*batch_size: (i+1)*batch_size]:
        data, label = train_loader.dataset[idx]
        z.append(data)
        label_z.append(label)
    z = torch.stack(z)
    label_z = torch.tensor(label_z)
    z = z.to(device)
    x = x.to(device)
    _, x_feature = model(x)
    z_calibrated = calibrate_feature(model, z, x_feature, n_iters)
    
    xs.append(x.detach_())
    ys.append(y.detach_())
    zs.append(z.detach_())
    y_zs.append(label_z.detach_())
    zcs.append(z_calibrated.detach_())

    endtime = time.time()
    print("{}, remaing time: {:.2f} h".format(i, (len(train_loader) - i)*(endtime - start_time)/3600))
xs = torch.stack(xs)
ys = torch.cat(ys)
y_zs = torch.cat(y_zs)
zs = torch.stack(zs)
zcs = torch.stack(zcs)
torch.save(xs, "output/learning_from_ce/xs.pt")
torch.save(ys, "output/learning_from_ce/ys.pt")
torch.save(y_zs, "output/learning_from_ce/y_zs.pt")
torch.save(zs, "output/learning_from_ce/zs.pt")
torch.save(zcs, "output/learning_from_ce/zcs.pt")


