import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn

import precision_profiler
import numpy as np

def quantizeWeight(W,r,BW): #r the range should be dividing W and multiplying Wq but ya.. not working so let's skip that
    Wq = np.minimum(np.round(W/r*np.power(2.0,BW-1.0))*np.power(2.0,1.0-BW),1.0-np.power(2.0,1.0-BW))*r
    return Wq

activation_dr= np.load("activation_dynamic_range.npy") 
weight_dr=np.load("weight_dynamic_range.npy")

class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter:
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = ProgressMeter._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def print2(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    @classmethod
    def _get_batch_fmtstr(cls, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


class Net(nn.Module):
    def __init__(self, num_classes=10):
        super(Net, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 192, kernel_size=3, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.fc_layers = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

        self.nclass = num_classes

    def forward(self, x):
        #conv_features = self.features(x)
        in_x = x
        for layer_idx,f in enumerate(self.features):
            weight_layers = [0,3,6,8,10,14,17,19]
            if layer_idx in weight_layers:
                #just need to quantize before and after weight layers
                #r is the dynamic range of the values
                r_idx =  weight_layers.index(layer_idx)
                r_w = weight_dr[r_idx]
                r_a = activation_dr[r_idx]
                print("r_idx,r_w,r_a",r_idx,r_w,r_a)
                m=64
                
                #quantize wieghts before weight layer
                in_x = quantizeWeight(in_x,r_w,m);#torch.from_numpy(inp_quant).float()
                #print("in weights postquant:",in_x[0,0,0])

                
                #run the layer
                out_x = f(in_x)
                
                #quantize activations after weight layer
                #out = out_x.detach().numpy()
                #lvls = UniformLevels(r_a,m, 'a')
                #out_quant = quantize(out, lvls)
         #       out_x = quantizeWeight(out_x,r_a,m) #torch.from_numpy(out_quant).float()
                #print("out_x post quant", out_x.size())

                in_x = out_x
            else:
                #no need to quantize if it's not a wieght layer
                out_x = f(in_x)
                in_x = out_x
                
        conv_features = out_x
        
        flatten = conv_features.view(conv_features.size(0), -1)
        
        #do the same thing as above here to quantize these layers
        fc = self.fc_layers(flatten) #this calls x(the input) with fc_layers () Sequential
        return fc
        flatten = conv_features.view(conv_features.size(0), -1)
        fc = self.fc_layers(flatten)
        return fc


def get_datasets(*args, **kwargs):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )

    trainset = torchvision.datasets.CIFAR10(train=True, transform=transform, *args, **kwargs)
    testset = torchvision.datasets.CIFAR10(train=False, transform=transform, *args, **kwargs)
    return trainset, testset


def get_dataloaders(trainset, testset, batch_size=100, num_worker=4):
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_worker)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_worker)

    return trainloader, testloader


def get_model(model_src_path, device='cpu'):
    model = Net(num_classes=10)
    state_dict = torch.load(model_src_path, map_location=device)
    model.load_state_dict(state_dict)
    return model


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        if type(output) is tuple:
            _, _, output = output
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res, pred[0, :]


def eval_single_batch_compute(x, y, model):
    output = model(x)
    accs, predictions = accuracy(output, y, topk=(1,))
    acc = accs[0]
    return acc, predictions


def eval_model(model, dataloader, print_acc=False, device='cpu', log_update_feq=20):
    model.eval()
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(dataloader),
        [top1],
        prefix='Evaluating Batch'
    )

    with torch.no_grad():
        for idx, data in enumerate(dataloader):
            x, y = data
            x.requires_grad = True
            x = x.to(device)
            y = y.to(device)
            n_data = y.size(0)

            acc, predictions = eval_single_batch_compute(x, y, model)

            top1.update(acc.item(), n_data)
            if idx % log_update_feq == log_update_feq - 1:
                progress.print2(idx + 1)

        if print_acc:
            print(' * Acc@1 {top1.avg:.3f}'.format(top1=top1))

    return top1.avg


def feedforward():
    device = 'cpu'
    print('using device:', device)
    trainset, testset = get_datasets(root='./data', download=True)
    _, testloader = get_dataloaders(trainset, testset, batch_size=100, num_worker=16)

    model_src_path = 'model.tar'  # todo you need to set the path to downloaded model !!
    model = get_model(model_src_path, device)
    model = model.to(device)
    eval_model(model, testloader, print_acc=True, device=device)


def compute_precision_offsets():
    device = 'cuda'
    print('using device:', device)
    trainset, testset = get_datasets(root='./data', download=True)
    trainloader, testloader = get_dataloaders(trainset, testset, batch_size=500, num_worker=32)

    model_src_path = 'model.tar'
    model = get_model(model_src_path, device)
    model = model.to(device)
    precision_profiler.GradCollect.retain_model_grad(model)
    precision_profiler.GradCollect.retain_inputs_grad(model)

    wg, ag = precision_profiler.get_noise_gains(model, trainloader, device)
    wg, ag, least_gain = precision_profiler.get_normalized_noise_gains(wg, ag)
    print(precision_profiler.GradCollect.weight_range_collect)
    print(precision_profiler.GradCollect.activation_range_collect)
    w_offsets, a_offsets = precision_profiler.get_precision_offsets(wg, ag, least_gain)
    print(w_offsets, a_offsets)
    np.save(arr=w_offsets,file='weight_offsets.npy')
    np.save(arr=a_offsets,file='activation_offsets.npy')
    np.save(arr=precision_profiler.GradCollect.activation_range_collect, file='activation_dynamic_range.npy')
    np.save(arr=np.array([v for k, v in precision_profiler.GradCollect.weight_range_collect]), file='weight_dynamic_range.npy')


if __name__ == '__main__':
    feedforward()