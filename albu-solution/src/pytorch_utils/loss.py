import torch
eps = 1

def dice_round(preds, trues, is_average=True):
    preds = torch.round(preds)
    return dice(preds, trues, is_average=is_average)


def dice(preds, trues, weight=None, is_average=True):
    num = preds.size(0)
    preds = preds.view(num, -1)
    trues = trues.view(num, -1)
    if weight is not None:
        w = torch.autograd.Variable(weight).view(num, -1)
        preds = preds * w
        trues = trues * w
    intersection = (preds * trues).sum(1)
    scores = (2. * intersection + eps) / (preds.sum(1) + trues.sum(1) + eps)

    score = scores.sum()
    if is_average:
        score /= num
    return torch.clamp(score, 0., 1.)

