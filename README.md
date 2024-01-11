# CASCADE

Official Pytorch implementation of [Medical Image Segmentation via Cascaded Attention Decoding, WACV 2023](https://openaccess.thecvf.com/content/WACV2023/html/Rahman_Medical_Image_Segmentation_via_Cascaded_Attention_Decoding_WACV_2023_paper.html). <br> [GAIN 2023 best poster award] (https://sites.utexas.edu/gain/previous-winners/prev-gain-2023-posterwinners)
<br>
[Md Mostafijur Rahman](https://github.com/mostafij-rahman), [Radu Marculescu](https://radum.ece.utexas.edu/)
<p>The University of Texas at Austin</p>


## Usage:
### Recommended environment:
```
Python 3.8
Pytorch 1.11.0
torchvision 0.12.0
```
Please use ```pip install -r requirements.txt``` to install the dependencies.

### Data preparation:
- **Synapse Multi-organ dataset:**
Sign up in the [official Synapse website](https://www.synapse.org/#!Synapse:syn3193805/wiki/89480) and download the dataset. Then split the 'RawData' folder into 'TrainSet' (18 scans) and 'TestSet' (12 scans) following the [TransUNet's](https://github.com/Beckschen/TransUNet/blob/main/datasets/README.md) lists and put in the './data/synapse/Abdomen/RawData/' folder. Finally, preprocess using ```python ./utils/preprocess_synapse_data.py``` or download the [preprocessed data](https://drive.google.com/file/d/1tGqMx-E4QZpSg2HQbVq5W3KSTHSG0hjK/view?usp=share_link) and save in the './data/synapse/' folder. 
Note: If you use the preprocessed data from [TransUNet](https://drive.google.com/drive/folders/1ACJEoTp-uqfFJ73qS3eUObQh52nGuzCd), please make necessary changes (i.e., remove the code segment (line# 88-94) to convert groundtruth labels from 14 to 9 classes) in the utils/dataset_synapse.py. 

- **ACDC dataset:**
Download the preprocessed ACDC dataset from [Google Drive of MT-UNet](https://drive.google.com/file/d/13qYHNIWTIBzwyFgScORL2RFd002vrPF2/view) and move into './data/ACDC/' folder.

- **Polyp datasets:**
Download training and testing datasets [Google Drive](https://drive.google.com/file/d/1pFxb9NbM8mj_rlSawTlcXG1OdVGAbRQC/view?usp=sharing) and move them into './data/polyp/'.


### Pretrained model:
You should download the pretrained PVTv2 model from [Google Drive](https://drive.google.com/drive/folders/1Eu8v9vMRvt-dyCH0XSV2i77lAd62nPXV?usp=sharing), and then put it in the './pretrained_pth/pvt/' folder for initialization. 

Download Google pretrained ViT models (R50-ViT-B_16, ViT-B_16, ...) from [Google Cloud](https://console.cloud.google.com/storage/browser/vit_models/imagenet21k) or use ```wget https://storage.googleapis.com/vit_models/imagenet21k/{MODEL_NAME}.npz```, and then put them in the './pretrained_pth/vit/imagenet21k/' folder for initialization. 

### Training:
```
cd into CASCADE
```
For Polyp training run ```CUDA_VISIBLE_DEVICES=0 python -W ignore train_polyp.py``` 

For Synapse Multi-organ training run ```CUDA_VISIBLE_DEVICES=0 python -W ignore train_synapse.py```

For ACDC training run ```CUDA_VISIBLE_DEVICES=0 python -W ignore train_ACDC.py```

### Testing:
```
cd into CASCADE 
```
For Polyp testing run ```CUDA_VISIBLE_DEVICES=0 python -W ignore test_polyp.py``` 

For Synapse Multi-organ testing run ```CUDA_VISIBLE_DEVICES=0 python -W ignore test_synapse.py```

For ACDC testing run ```CUDA_VISIBLE_DEVICES=0 python -W ignore test_ACDC.py```

## Acknowledgement
We are very grateful for these excellent works [PraNet](https://github.com/DengPingFan/PraNet), [Polyp-PVT](https://github.com/DengPingFan/Polyp-PVT) and [TransUNet](https://github.com/Beckschen/TransUNet), which have provided the basis for our framework.

## Citations

``` 
@InProceedings{Rahman_2023_WACV,
    author    = {Rahman, Md Mostafijur and Marculescu, Radu},
    title     = {Medical Image Segmentation via Cascaded Attention Decoding},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    month     = {January},
    year      = {2023},
    pages     = {6222-6231}
} 
```
