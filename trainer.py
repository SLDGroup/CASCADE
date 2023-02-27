import argparse
import logging
import os
import random
import sys
import time
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms

from utils.dataset_synapse import Synapse_dataset, RandomGenerator
from utils.utils import one_hot_encoder
from utils.utils import DiceLoss
from utils.utils import val_single_volume

def inference(args, model, best_performance):
    db_test = Synapse_dataset(base_dir=args.volume_path, split="test_vol", list_dir=args.list_dir, nclass=args.num_classes)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    logging.info("{} test iterations per epoch".format(len(testloader)))
    model.eval()
    metric_list = 0.0
    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        h, w = sampled_batch["image"].size()[2:]
        image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
        metric_i = val_single_volume(image, label, model, classes=args.num_classes, patch_size=[args.img_size, args.img_size],
                                      case=case_name, z_spacing=args.z_spacing)
        metric_list += np.array(metric_i)
    metric_list = metric_list / len(db_test)
    performance = np.mean(metric_list, axis=0)
    logging.info('Testing performance in val model: mean_dice : %f, best_dice : %f' % (performance, best_performance))
    return performance

def trainer_synapse(args, model, snapshot_path):
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu

    db_train = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train", nclass=args.num_classes,
                               transform=transforms.Compose(
                                   [RandomGenerator(output_size=[args.img_size, args.img_size])]))
    print("The length of train set is: {}".format(len(db_train)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)

    #optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=0.0001)
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            
            p1, p2, p3, p4 = model(image_batch) # forward
            
            outputs = p1 + p2 + p3 + p4 # additive output aggregation
            
            loss_ce1 = ce_loss(p1, label_batch[:].long())
            loss_ce2 = ce_loss(p2, label_batch[:].long())
            loss_ce3 = ce_loss(p3, label_batch[:].long())
            loss_ce4 = ce_loss(p4, label_batch[:].long())
            loss_dice1 = dice_loss(p1, label_batch, softmax=True)
            loss_dice2 = dice_loss(p2, label_batch, softmax=True)
            loss_dice3 = dice_loss(p3, label_batch, softmax=True)
            loss_dice4 = dice_loss(p4, label_batch, softmax=True)
            
            
            loss_p1 = 0.3 * loss_ce1 + 0.7 * loss_dice1
            loss_p2 = 0.3 * loss_ce2 + 0.7 * loss_dice2
            loss_p3 = 0.3 * loss_ce3 + 0.7 * loss_dice3
            loss_p4 = 0.3 * loss_ce4 + 0.7 * loss_dice4
            
            alpha, beta, gamma, zeta = 1., 1., 1., 1.
            loss = alpha * loss_p1 + beta * loss_p2 + gamma * loss_p3 + zeta * loss_p4 # current setting is for additive aggregation.
           
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9 # we did not use this
            lr_ = base_lr
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            

            if iter_num % 20 == 0:
                logging.info('iteration %d, epoch %d : loss : %f, lr: %f' % (iter_num, epoch_num, loss.item(), lr_))
                image = image_batch[1, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction', outputs[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)
     
        
        logging.info('iteration %d, epoch %d : loss : %f, lr: %f' % (iter_num, epoch_num, loss.item(), lr_))
        performance = inference(args, model, best_performance)
        
        save_interval = 50

        if(best_performance <= performance):
            best_performance = performance
            save_mode_path = os.path.join(snapshot_path, 'best.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            
        if (epoch_num + 1) % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            break

    writer.close()
    return "Training Finished!"
