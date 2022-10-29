import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
from scipy import misc
import cv2

from lib.networks import PVT_CASCADE
from utils.dataloader import test_dataset

if __name__ == '__main__':
    method_name = 'PolypPVT-CASCADE'
    parser = argparse.ArgumentParser()
    parser.add_argument('--testsize', type=int, default=352, help='testing size')
    parser.add_argument('--pth_path', type=str, default='./model_pth/'+method_name+'/'+method_name+'.pth')
    opt = parser.parse_args()
    
    #torch.cuda.set_device(0)  # set your gpu device
    
    model = PVT_CASCADE()
    model.cuda()
    model.load_state_dict(torch.load(opt.pth_path))
    
    model.eval()
    
    for _data_name in ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB']:

        ##### put data_path here #####
        data_path = './data/polyp/TestDataset/{}'.format(_data_name)
        
        ##### save_path #####
        save_path = './result_map/'+method_name+'/{}/'.format(_data_name)

        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        print('Evaluating ' + data_path)
        
        image_root = '{}/images/'.format(data_path)
        gt_root = '{}/masks/'.format(data_path)
        num1 = len(os.listdir(gt_root))
        test_loader = test_dataset(image_root, gt_root, 352)
        DSC = 0.0
        JACARD = 0.0
        preds = []
        gts = []
        for i in range(num1):
            image, gt, name = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.cuda()
            
            res1, res2, res3, res4 = model(image) # forward
            
            # eval Dice
            res = F.upsample(res1 + res2 + res3 + res4, size=gt.shape, mode='bilinear', align_corners=False)

            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            cv2.imwrite(save_path+name, res*255)        
            
            input = np.where(res >= 0.5, 1, 0)
            target = np.where(np.array(gt) >= 0.5, 1, 0)
            
            preds.append(input)
            gts.append(gt)
            
            N = gt.shape
            smooth = 1
            input_flat = np.reshape(input, (-1))
            target_flat = np.reshape(target, (-1))
            intersection = (input_flat * target_flat)
            union = input_flat + target_flat - intersection
            
            jacard = ((np.sum(intersection)+smooth)/(np.sum(union)+smooth))
            jacard = '{:.4f}'.format(jacard)
            jacard = float(jacard)
            JACARD += jacard
            
            dice = (2 * intersection.sum() + smooth) / (input.sum() + target.sum() + smooth)
            dice = '{:.4f}'.format(dice)
            dice = float(dice)
            DSC += dice
            
        print('*****************************************************')
        print('Dice Score: ' + str(DSC/num1))
        print('Jacard Score: ' + str(JACARD/num1))
        
        print(_data_name, 'Finish!')
        print('*****************************************************')
