# DAWN\+
DAWN\+: Wavelet-based Image Deraining Meets Direction-aware Attention and Mutual Representation (arxiv)

[Kui Jiang](https://scholar.google.com/citations?user=AbOLE9QAAAAJ&hl), [Junjun Jiang](https://scholar.google.com/citations?user=WNH2_rgAAAAJ&hl=zh-CN), [Zheng Wang](https://scholar.google.com/citations?user=-WHTbpUAAAAJ&hl=zh-CN), [Zihan Geng]() and [Chia-Wen Lin](https://scholar.google.com/citations?user=fXN3dl0AAAAJ&hl=zh-CN)

**Paper**: [DAWN+: Wavelet-based Image Deraining Meets Direction-aware Attention and Mutual Representation](https://scholar.google.com/citations?view_op=view_citation&hl=zh-CN&user=AbOLE9QAAAAJ&sortby=pubdate&citation_for_view=AbOLE9QAAAAJ:WbkHhVStYXYC)


## Installation
The model is built in PyTorch 1.1.0 and tested on Ubuntu 16.04 environment (Python3.7, CUDA9.0, cuDNN7.5).

For installing, follow these intructions
```
conda create -n pytorch1 python=3.7
conda activate pytorch1
conda install pytorch=1.1 torchvision=0.3 cudatoolkit=9.0 -c pytorch
pip install matplotlib scikit-image opencv-python yacs joblib natsort h5py tqdm
```

## Quick Test

To test the pre-trained deraining model on your own images, run 
```
python test.py  
```

## Training and Evaluation

### Training
- Download the [Datasets](Datasets/README.md)

- Train the model with default arguments by running

```
python train_DAWN_PLUS.py
```


### Evaluation

1. Download the [model]() and place it in `./pretrained_models/`

2. Download test datasets (Test100, Rain100H, Rain100L, Test1200, Test2800) from [here](https://drive.google.com/drive/folders/1PDWggNh8ylevFmrjo-JEvlmqsDlWWvZs?usp=sharing) and place them in `./Datasets/Synthetic_Rain_Datasets/test/`

3. Run
```
python test.py
```

#### To reproduce PSNR/SSIM scores of the paper, run
```
evaluate_PSNR_SSIM.m 
```

## Results
Experiments are performed for different image processing tasks including, image deraining, image dehazing and low-light image enhancement.

## Acknowledgement
Code borrows from [MPRNet](https://github.com/swz30/MPRNet) by [Syed Waqas Zamir](https://scholar.google.es/citations?user=WNGPkVQAAAAJ&hl=en). Thanks for sharing !

## Citation
If you use DAWN and DAWN\+, please consider citing:

    @article{jiang2023dawn,
        title={Dawn: Direction-aware attention wavelet network for image deraining},
        author={Jiang, Kui and Liu, Wenxuan and Wang, Zheng and Zhong, Xian and Jiang, Junjun and Lin, Chia-Wen},
        journal={Proceedings of the 31st ACM International Conference on Multimedia}, 
        pages={7065--7074},
        year={2023}
    }

    @article{jiang2023dawnPLUS,
        title={DAWN+: Wavelet-based Image Deraining Meets Direction-aware Attention and Mutual Representation},
        author={Jiang, Kui and Jiang, Junjun and Wang, Zheng and Gen, Zihan and Liu, Xianming},
        journal={arxiv}, 
        year={2024}
    }

## Contact
Should you have any question, please contact Kui Jiang (kuijiang@whu.edu.cn)
