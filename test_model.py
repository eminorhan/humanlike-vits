import argparse
import torch
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from utils import load_model


def get_args_parser():
    parser = argparse.ArgumentParser('Test a model on ImageNet val data', add_help=False)
    parser.add_argument('--model_id', default='vith14@476_1.0_1', type=str, help='Model identifier')
    parser.add_argument('--input_size', default=224, type=int, help='input image size')
    parser.add_argument('--val_data_path', default='', type=str)
    parser.add_argument('--device', default='cuda', help='device to use for testing')
    parser.add_argument('--batch_size', default=128, type=int, help='batch size for eval')
    parser.add_argument('--num_workers', default=8, type=int)

    return parser


def accuracy(output, target, topk=(1,)):
    """computes top-k accuracy for specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[:k].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]


def main(args):
    model = load_model(args.model_id, finetuned=True)
    print('Model:', model)

    device = torch.device(args.device)
    model.to(device)  # move model to device
    
    # prepare val data
    val_transform = Compose([Resize(args.input_size + 32, interpolation=3), CenterCrop(args.input_size), ToTensor(), Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    val_dataset = ImageFolder(args.val_data_path, transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    targets, outputs = [], []

    with torch.no_grad():
        # switch to eval mode
        model.eval()

        for _, (inp, target) in enumerate(val_loader):
            inp = inp.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            
            output = model(inp)
            
            targets.append(target)
            outputs.append(output)

        targets = torch.cat(targets)
        outputs = torch.cat(outputs)
        acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
        print('* Acc@1 {acc1:.3f} Acc@5 {acc5:.3f}'.format(acc1=acc1, acc5=acc5))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    print(args)
    main(args)