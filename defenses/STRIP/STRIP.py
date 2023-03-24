import torch
import os
import torchvision
import numpy as np
import copy
import cv2
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision import transforms
from config import get_argument

import sys

sys.path.insert(0, "../..")
from classifier_models import PreActResNet18, ResNet18
from utils.dataloader import get_dataloader, get_dataset
from utils.utils import progress_bar
from networks.models import NetC_MNIST, Normalizer, Denormalizer
from networks.dba import MnistNet, ResNet18
from networks.lira import VGG as vgg9, Net
from networks.resnet_tinyimagenet import resnet18

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
IMAGENET_MIN  = ((np.array([0,0,0]) - np.array(IMAGENET_DEFAULT_MEAN)) / np.array(IMAGENET_DEFAULT_STD)).min()
IMAGENET_MAX  = ((np.array([1,1,1]) - np.array(IMAGENET_DEFAULT_MEAN)) / np.array(IMAGENET_DEFAULT_STD)).max()

class Normalize:
    def __init__(self, opt, expected_values, variance):
        self.n_channels = opt.input_channel
        self.expected_values = expected_values
        self.variance = variance
        assert self.n_channels == len(self.expected_values)

    def __call__(self, x):
        x_clone = x.clone()
        for channel in range(self.n_channels):
            x_clone[:, :, channel] = (x[:, :, channel] - self.expected_values[channel]) / self.variance[channel]
        return x_clone


class Denormalize:
    def __init__(self, opt, expected_values, variance):
        self.n_channels = opt.input_channel
        self.expected_values = expected_values
        self.variance = variance
        assert self.n_channels == len(self.expected_values)

    def __call__(self, x):
        x_clone = x.clone()
        for channel in range(self.n_channels):
            x_clone[:, :, channel] = x[:, :, channel] * self.variance[channel] + self.expected_values[channel]
        return x_clone


class STRIP:
    def _superimpose(self, background, overlay):
        output = cv2.addWeighted(background, 1, overlay, 1, 0)
        if len(output.shape) == 2:
            output = np.expand_dims(output, 2)
        return output

    def _get_entropy(self, background, dataset, classifier):
        entropy_sum = [0] * self.n_sample
        x1_add = [0] * self.n_sample
        index_overlay = np.random.randint(0, len(dataset), size=self.n_sample)
        for index in range(self.n_sample):
            add_image = self._superimpose(background, dataset[index_overlay[index]][0])
            add_image = self.normalize(add_image)
            x1_add[index] = add_image

        py1_add = classifier(torch.stack(x1_add).to(self.device))
        py1_add = torch.sigmoid(py1_add).cpu().numpy()
        entropy_sum = -np.nansum(py1_add * np.log2(py1_add))
        return entropy_sum / self.n_sample

    def _get_denormalize(self, opt):
        if opt.dataset == "cifar10":
            denormalizer = Denormalize(opt, [0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
        elif opt.dataset == "mnist":
            denormalizer = Denormalize(opt, [0.5], [0.5])
        elif opt.dataset == "gtsrb" or opt.dataset == "celeba":
            denormalizer = None
        elif opt.dataset in ['timagenet', 'tiny-imagenet32']:
            denormalizer = Denormalize(opt, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        else:
            raise Exception("Invalid dataset")
        return denormalizer

    def _get_normalize(self, opt):
        if opt.dataset == "cifar10":
            normalizer = Normalize(opt, [0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
        elif opt.dataset == "mnist":
            normalizer = Normalize(opt, [0.5], [0.5])
        elif opt.dataset == "gtsrb" or opt.dataset == "celeba":
            normalizer = None
        elif opt.dataset in ['timagenet', 'tiny-imagenet32']:
            normalizer = Normalize(opt, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        else:
            raise Exception("Invalid dataset")
        if normalizer:
            transform = transforms.Compose([transforms.ToTensor(), normalizer])
        else:
            transform = transforms.ToTensor()
        return transform

    def __init__(self, opt):
        super().__init__()
        self.n_sample = opt.n_sample
        self.normalizer = self._get_normalize(opt)
        self.denormalizer = self._get_denormalize(opt)
        self.device = opt.device

    def normalize(self, x):
        if self.normalizer:
            x = self.normalizer(x)
        return x

    def denormalize(self, x):
        if self.denormalizer:
            x = self.denormalizer(x)
        return x

    def __call__(self, background, dataset, classifier):
        return self._get_entropy(background, dataset, classifier)

def get_clip_image(dataset="cifar10"):
    if dataset in ['tiny-imagenet', 'tiny-imagenet32']:
        def clip_image(x):
            return torch.clamp(x, IMAGENET_MIN, IMAGENET_MAX)
    elif dataset == 'cifar10':
        def clip_image(x):
            return torch.clamp(x, IMAGENET_MIN, IMAGENET_MAX)
    elif dataset == 'mnist':
        def clip_image(x):
            return torch.clamp(x, -1.0, 1.0)
    elif dataset == 'gtsrb':
        def clip_image(x):
            return torch.clamp(x, IMAGENET_MIN, IMAGENET_MAX)
    else:
        raise Exception(f'Invalid dataset: {dataset}')
    return clip_image     

def create_trigger_model(dataset, device="cpu", attack_model=None):
    """ Create trigger model for LIRA """
    if dataset == 'cifar10':
        from attack_models.unet import UNet
        atkmodel = UNet(3).to(device)
    elif dataset == 'mnist':
        from attack_models.autoencoders import MNISTAutoencoder as Autoencoder
        atkmodel = Autoencoder().to(device)
    elif dataset == 'timagenet' or dataset == 'tiny-imagenet32' or dataset == 'gtsrb':
        if attack_model is None:
            from attack_models.autoencoders import Autoencoder
            atkmodel = Autoencoder().to(device)
        elif attack_model == 'unet':
            from attack_models.unet import UNet
            atkmodel = UNet(3).to(device)
    else:
        raise Exception(f'Invalid atk model {dataset}')
    return atkmodel

def create_backdoor(inputs, identity_grid, noise_grid, opt):
    bs = inputs.shape[0]
    grid_temps = (identity_grid + opt.s * noise_grid / opt.input_height) * opt.grid_rescale
    grid_temps = torch.clamp(grid_temps, -1, 1)

    bd_inputs = F.grid_sample(inputs, grid_temps.repeat(bs, 1, 1, 1), align_corners=True)
    return bd_inputs

def get_poisoned_data(inputs, atkmodel, opt):
    # TODO: this is for LIRA only, Dung will update it later
    clip_image=get_clip_image(opt.dataset)
    bs = inputs.shape[0]
    noise = atkmodel(inputs)
    atkdata = clip_image(inputs + noise)
    return atkdata

# def superimpose(inputs, background, bs):
#     index_overlay = np.random.randint(40000,49999, size=bs)
#     x1_add = [0] * bs
#     for x in range(bs):
#         x_background = x_train[j+26000] 
#         x1_add[x] = cv2.addWeighted()
def strip(opt, mode="clean"):
    # print(f"mode is: {mode}")
    # Prepare pretrained classifier
    if opt.dataset == "mnist":
        netC = NetC_MNIST().to(opt.device)
    elif opt.dataset == "cifar10" or opt.dataset == "gtsrb":
        netC = PreActResNet18(num_classes=opt.num_classes).to(opt.device)
    elif opt.dataset == "celeba":
        netC = ResNet18().to(opt.device)
    elif opt.dataset == "timagenet":
        netC = resnet18(num_classes=200).to(opt.device)
    else:
        raise Exception("Invalid dataset")

    # Load pretrained model
    # mode = opt.attack_mode
    opt.ckpt_folder = os.path.join(opt.checkpoints, opt.dataset)
    opt.ckpt_path = os.path.join(opt.ckpt_folder, "{}_{}_morph.pth.tar".format(opt.dataset, mode))
    opt.log_dir = os.path.join(opt.ckpt_folder, "log_dir")

    # state_dict = torch.load(opt.ckpt_path)
    # netC.load_state_dict(state_dict["netC"])
    
    # TODO: Dung will modify it later
    ckpt_path = os.path.join(
        opt.checkpoints, opt.dataset, "{}_{}_morph.pth.tar".format(opt.dataset, opt.attack_mode)
    )

    if opt.backdoor_type == "dba":
        if opt.dataset == "mnist":
            ckpt_path = os.path.join(opt.checkpoints, opt.dataset, "model_last.pt.tar")
            netC = MnistNet(name='Local')
        elif opt.dataset == "cifar10":
            ckpt_path = os.path.join(opt.checkpoints, opt.dataset, "model_last.pt.tar.epoch_400")
            netC = ResNet18(name='Local')                
    elif opt.backdoor_type == "lira":
        if opt.dataset == "mnist":
            ckpt_path = os.path.join(opt.checkpoints, opt.dataset, "model_last.pt.tar")
            netC = Net()
        elif opt.dataset == "cifar10":
            ckpt_path = os.path.join(opt.checkpoints, opt.dataset, "lira_cifar10_vgg9_0.03.pt")
            # classifier = ResNet18(name='Local')     
            netC = vgg9.VGG('VGG9')
        elif opt.dataset == "timagenet":
            ckpt_path = os.path.join(opt.checkpoints, opt.dataset, "lira_timagenet_vgg9_0.03.pt")
            # netC = vgg9.VGG('VGG9')
            netC = resnet18(num_classes=200)
        if opt.ckpt_file:
            ckpt_path = os.path.join(opt.checkpoints, opt.dataset, opt.ckpt_file)
        print("Loaded checkpoint path successfully")
    else:
        state_dict = torch.load(opt.ckpt_path)
        netC.load_state_dict(state_dict["netC"])
    atkmodel = None
    if opt.backdoor_type == "lira":
        # checkpoint_path = os.path.join(opt.checkpoints, opt.dataset, "atkmodel", "lira_cifar10_vgg9_unet.pt")
        checkpoint_path = os.path.join(opt.checkpoints, opt.dataset, "atkmodel", opt.ckpt_file)

        atkmodel = create_trigger_model(opt.dataset, opt.device)
        atkmodel.load_state_dict(torch.load(checkpoint_path))

    # print(f"atkmodel: {atkmodel}")
    state_dict = torch.load(ckpt_path)
    # classifier.load_state_dict(state_dict["netC"])
    if opt.backdoor_type != "lira":
        netC.load_state_dict(state_dict["state_dict"])         
    else:
        netC.load_state_dict(state_dict)  
    # print(f"mode 2 is: {mode}")
    if mode != "clean" and opt.backdoor_type != "lira":
        identity_grid = state_dict["identity_grid"]
        noise_grid = state_dict["noise_grid"]
        

    netC.requires_grad_(False)
    netC.eval()
    netC.to(opt.device)
    if atkmodel:
        atkmodel.requires_grad_(False)
        atkmodel.eval()
        atkmodel.to(opt.device)

    # Prepare test set
    testset = get_dataset(opt, train=False)
    opt.bs = opt.n_test
    test_dataloader = get_dataloader(opt, train=False)
    test_dataloader_cp = copy.deepcopy(test_dataloader)
    denormalizer = Denormalizer(opt)

    # STRIP detector
    strip_detector = STRIP(opt)

    # Entropy list
    list_entropy_trojan = []
    list_entropy_benign = []

    if mode == "attack":
        # Testing with perturbed data
        print("Testing with backdoor data !!!!")
        inputs, targets = next(iter(test_dataloader))
        inputs = inputs.to(opt.device)
        clean_inputs = copy.deepcopy(inputs)
        if opt.backdoor_type != "lira":
            bd_inputs = create_backdoor(inputs, identity_grid, noise_grid, opt)
            bd_inputs = denormalizer(bd_inputs) * 255.0
        else:
            bd_inputs = get_poisoned_data(inputs, atkmodel, opt)

        bd_inputs = bd_inputs.detach().cpu().numpy()
        bd_inputs = np.clip(bd_inputs, 0, 255).astype(np.uint8).transpose((0, 2, 3, 1))
        for index in range(opt.n_test):
            background = bd_inputs[index]
            entropy = strip_detector(background, testset, netC)
            list_entropy_trojan.append(entropy)
            progress_bar(index, opt.n_test)
        # print(f"list_entropy_trojan: {list_entropy_trojan}")


        # Testing with clean data
        for index in range(opt.n_test):
            background, _ = testset[index]
            # background = clean_inputs[index]
            entropy = strip_detector(background, testset, netC)
            list_entropy_benign.append(entropy)
    else:
    # if True:
        # Testing with clean data
        print("Testing with clean data !!!!")
        for index in range(opt.n_test):
            background, _ = testset[index]
            entropy = strip_detector(background, testset, netC)
            list_entropy_benign.append(entropy)
            progress_bar(index, opt.n_test)

    return list_entropy_trojan, list_entropy_benign


def main():
    opt = get_argument().parse_args()
    if opt.dataset == "mnist":
        opt.input_height = 28
        opt.input_width = 28
        opt.input_channel = 1
        opt.num_classes = 10
    elif opt.dataset == "cifar10":
        opt.input_height = 32
        opt.input_width = 32
        opt.input_channel = 3
        opt.num_classes = 10
    elif opt.dataset == "gtsrb":
        opt.input_height = 32
        opt.input_width = 32
        opt.input_channel = 3
        opt.num_classes = 43
    elif opt.dataset == "celeba":
        opt.input_height = 64
        opt.input_width = 64
        opt.input_channel = 3
        opt.num_classes = 8
    elif opt.dataset == "timagenet":
        opt.input_height = 64
        opt.input_width = 64
        opt.input_channel = 3
        opt.num_classes = 200
    else:
        raise Exception("Invalid dataset")

    if "2" in opt.attack_mode:
        mode = "attack"
    else:
        mode = "clean"

    mode = "attack"
    lists_entropy_trojan = []
    lists_entropy_benign = []
    for test_round in range(opt.test_rounds):
        list_entropy_trojan, list_entropy_benign = strip(opt, mode)
        lists_entropy_trojan += list_entropy_trojan
        lists_entropy_benign += list_entropy_benign


    # Plot result here
    bins = 30
    print(f"lists_entropy_benign: {lists_entropy_benign} \n lists_entropy_trojan: {lists_entropy_trojan}")
    plt.hist(lists_entropy_benign, bins, weights=np.ones(len(lists_entropy_benign)) / len(lists_entropy_benign), alpha=1, label='without trojan')
    plt.hist(lists_entropy_trojan, bins, weights=np.ones(len(lists_entropy_trojan)) / len(lists_entropy_trojan), alpha=1, label='with trojan')
    plt.legend(loc='upper right', fontsize = 20)
    plt.ylabel('Probability (%)', fontsize = 20)
    plt.title('normalized entropy', fontsize = 20)
    plt.tick_params(labelsize=20)

    fig1 = plt.gcf()
    plt.show()
    fig1.savefig('EntropyDNNDist_T2.jpg')# save the fig as pdf file
    # fig1.savefig('EntropyDNNDist_T3.svg')# save the fig as pdf file

    # Save result to file
    result_dir = os.path.join(opt.results, opt.dataset)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    result_path = os.path.join(result_dir, opt.attack_mode)
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    result_path = os.path.join("{}_{}_{}_output.txt".format(opt.dataset, opt.attack_mode, opt.backdoor_type))

    with open(result_path, "w+") as f:
        for index in range(len(lists_entropy_trojan)):
            if index < len(lists_entropy_trojan) - 1:
                f.write("{} ".format(lists_entropy_trojan[index]))
            else:
                f.write("{}".format(lists_entropy_trojan[index]))

        f.write("\n")

        for index in range(len(lists_entropy_benign)):
            if index < len(lists_entropy_benign) - 1:
                f.write("{} ".format(lists_entropy_benign[index]))
            else:
                f.write("{}".format(lists_entropy_benign[index]))

    min_entropy = min(lists_entropy_trojan + lists_entropy_benign)

    # Determining
    print("Min entropy trojan: {}, Detection boundary: {}".format(min_entropy, opt.detection_boundary))
    if min_entropy < opt.detection_boundary:
        print("A backdoored model\n")
    else:
        print("Not a backdoor model\n")


if __name__ == "__main__":
    main()
