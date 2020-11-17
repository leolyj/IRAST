# IRAST
This is the code for the ECCV2020 paper "Semi-Supervised Crowd Counting via Self-Training on Surrogate Tasks" (Pytorch version)

## Prepare 
  1.1 Datasets can Found in:
  
   [ShanghaiTech](https://pan.baidu.com/s/1nuAYslz/)
   
   
   [UCF-QNRF]( https://drive.google.com/open?id=1fLZdOsOXlv2muNB_bXEW6t-IS9MRziL6)
   
   
   [WorldExpo10]( http://www.ee.cuhk.edu.hk/~xgwang/expo.html)
    
  1.2 Setting Runing Environment：
  
  Ubuntu 16.04
  
  
  Cuda 8.0
  
  
  python 2.7
  
  
  Pytorch 0.4.1
  
## Data Processing:
  follow the file "make_dataset.py" to produce the ground-truth density map (in this work, most images are unlabeled)
  
## Training the model:
  python train.py train.json val.json 0 0 to train your model
  
## Testing the model:
  python val.py 
  
  Notice the path of all files in these codes, you should modify them to suit your condition.
  
 ## Some Pre-trained Model in This Paper:
 ShanghaiTech PartA:[BaiduDisk](https://pan.baidu.com/s/1uYBtd9O0LfPUxveDmc7WQA )   password/code：2333 
 
 
 UCF-QNRF:[Baidudisk](https://pan.baidu.com/s/1TWsoIQZJBrGeMPnqlSQYCg)  password/code：2333 
 
 
 
 If you find the IRAST is useful, please cite our paper. Thank you!
 
 @inproceedings{liu2020semi,
  title={Semi-Supervised Crowd Counting via Self-Training on Surrogate Tasks},
  author={Liu, Yan and Liu, Lingqiao and Wang, Peng and Zhang, Pingping and Lei, Yinjie},
  booktitle={European Conference on Computer Vision},
  year={2020}
}
 
