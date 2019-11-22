import torch
from torch.autograd import Variable
import time
import os
import sys
import numpy as np
from utils import AverageMeter, calculate_accuracy

def train_epoch(epoch, data_loader, model, criterion, optimizer,
                epoch_logger, batch_logger):
    print('train at epoch {}'.format(epoch))

    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    # accuracies = AverageMeter()
    y_hat = []
    y = []
    end_time = time.time()
    for i, (clip, clip2, targets) in enumerate(data_loader):
        data_time.update(time.time() - end_time)

        targets = targets.cuda(async=True)
        
        clip = Variable(clip).cuda()
        clip2 = Variable(clip2).cuda()
        # todo
        targets = Variable(targets).float()

        outputs = model(clip, clip2)
        if len(y_hat) == 0:
            y_hat = list(outputs)
            y = list(targets)
        else:
            y_hat.extend(outputs)
            y.extend(targets)
        loss = criterion(outputs, targets)
        # loss += 0.001/np.corrcoef(outputs.cpu().detach().numpy(), targets.cpu().detach().numpy())[0,1]
        # vx = outputs - torch.mean(outputs)
        # vy = targets - torch.mean(targets)

        # loss = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
        # acc = calculate_accuracy(outputs, targets)

        losses.update(loss.data, clip.size(0))
        # accuracies.update(acc, inputs.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end_time)
        end_time = time.time()

        batch_logger.log({
            'epoch': epoch,
            'batch': i + 1,
            'iter': (epoch - 1) * len(data_loader) + (i + 1),
            'loss': losses.val,
            # 'acc': accuracies.val,
            'lr': optimizer.param_groups[0]['lr']
        })
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
        # torch.cuda.empty_cache()

    y = np.array(y,dtype=float)
    y_hat = np.array(y_hat,dtype=float)
    # print(y.shape, y_hat.shape)
    cor = np.corrcoef(y,y_hat)[0,1]
    print('train pearson:', cor)
    epoch_logger.log({
        'epoch': epoch,
        'loss': losses.avg,
        # 'acc': accuracies.avg,
        'lr': optimizer.param_groups[0]['lr'],
        'cor': cor
    })

    # if epoch % 20 == 0:
    #     save_file_path = os.path.join(opt.result_path,
    #                                   'save_{}.pth'.format(epoch))
    #     states = {
    #         'epoch': epoch + 1,
    #         'arch': opt.arch,
    #         'state_dict': model.state_dict(),
    #         'optimizer': optimizer.state_dict(),
    #     }
    #     torch.save(states, save_file_path)
