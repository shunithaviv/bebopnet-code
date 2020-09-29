import torch


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name='', fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

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


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t().type_as(target)
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def per_class_accuracy(output, target, n_classes):
    """Compute the accuracy per class"""
    batch_size = target.size(0)

    pred = torch.argmax(output, 1)
    pred = pred.unsqueeze(-1).type_as(target)
    pred_expanded = torch.zeros(batch_size, n_classes, device=output.device, dtype=target.dtype).scatter_(1, pred, 1)
    target_expanded = torch.zeros(batch_size, n_classes, device=output.device, dtype=target.dtype).scatter_(1, target,
                                                                                                            1)
    torch.stack((pred_expanded, target_expanded), )
    correct = torch.eq(torch.stack((pred_expanded, target_expanded)).sum(0), 2)

    total_target = target_expanded.sum(0)
    total_target[total_target == 0] = 1
    total_correct = correct.sum(0)

    total_correct = total_correct.float()
    total_target = total_target.float()
    accuracy = total_correct / total_target

    return accuracy
