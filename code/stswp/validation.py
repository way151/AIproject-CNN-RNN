import torch
from torch.autograd import Variable
import time
import sys

import numpy as np

from utils import AverageMeter, calculate_accuracy


def val_epoch(epoch, data_loader, model, criterion, logger):
    print('validation at epoch {}'.format(epoch))

    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    # accuracies = AverageMeter()

    end_time = time.time()
    y_hat = []
    y = []
    for i, (clip, clip2, targets) in enumerate(data_loader):
        data_time.update(time.time() - end_time)

        targets = targets.cuda(async=True)
        with torch.no_grad():
            clip = Variable(clip).cuda()
            clip2 = Variable(clip2).cuda()
            
            targets = Variable(targets).float()

            outputs = model(clip, clip2)


            if len(y_hat) == 0:
                y_hat = list(outputs)
                y = list(targets)
            else:
                y_hat.extend(outputs)
                y.extend(targets)

            loss = criterion(outputs, targets)
            # acc = calculate_accuracy(outputs, targets)

            losses.update(loss.data, clip.size(0))
            # accuracies.update(acc, inputs.size(0))

            batch_time.update(time.time() - end_time)
            end_time = time.time()
            if i % 10 == 0:  
                print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                  # 'Acc {acc.val:.3f} ({acc.avg:.3f})'
                      epoch,
                      i + 1,
                      len(data_loader),
                      batch_time=batch_time,
                      data_time=data_time,
                      loss=losses,
                      # acc=accuracies
                      ))
    y = np.array(y,dtype=float)
    y_hat = np.array(y_hat,dtype=float)
    cor = np.corrcoef(y,y_hat)[0,1]
    print('val pearson:', cor)
    print(y_hat)

    logger.log({'epoch': epoch, 'loss': losses.avg, 'cor': cor})
    return losses.avg
