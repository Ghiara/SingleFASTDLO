
<div align="center">
<h2> FASTDLO: Fast Deformable Linear Objects Instance Segmentation </h2>

 :page_with_curl:  [IEEE Xplore](https://ieeexplore.ieee.org/document/9830852)  :page_with_curl:	
</div>

### Demo



### Notation 
<div align="center">
[shema](shema.png)
</div>
The semantic segmentation module is inherited from above paper mentioned 'FASTDLO' algorithm, we modified this as it can only detect one single DLO with detection path, which can be used in the tracking algorithm e.g. CPD.

### Installation

Main dependencies:

The program runs on a Laptop with Ubuntu 22.04 LTS, with 32 GB RAM and Nvidia RTX 1650 mobile graphic card, envrionment management used miniconda, for more details about environment please refer the file 'requriements.txt'. Some key packages need to be installed please see below.
```
python (3.8)
pytorch (1.4.0)
opencv 
pillow 
scikit-image 
scipy 
shapely 
```

Installation:
```
pip install -e .
```

### Models' weights

Download the [weights](https://drive.google.com/file/d/1OVcro53E_8oJxRPHqGy619rBNoCD3rzT/view?usp=sharing) and place them inside a ```weights``` folder.


### Usage

import as a standard python package with 
```
from fastdlo.core import Pipeline
```

Then initialize the class 
``` 
p = Pipeline(checkpoint_siam=checkpoint_siam, checkpoint_seg=checkpoint_seg, img_w=IMG_W, img_h=IMG_H, colorRange=colorRange, is_interpolation=activate_interpolation)
```
where the ```p``` is Pipeline instance, ```checkpoint_*``` are the ckpt file paths (include file name), ```img_w``` is frame width, ```img_h``` is frame height, ```colorRange``` is HSV limitation that should be detected, in form of ```[((H_lower, S_lower, V_lower),(H_upper, S_upper, V_upper)),..]```, and ```is_interpolation``` is the boolean variable that indicates whether the interpolation functionality should be activated.

the inference can be obtained with 
```
img_out, dlo_mask_pointSet, dlo_path = p.run(source_img=color_image, mask_th=77)
```
where the ```img_out``` is the output masked image, ```dlo_mask_pointSet``` is the points array of mask in the image pixel unit [x,y], ```dlo_path``` is the points array of DLO detection path in the image pixel unit [x,y]

A example run file is prepared with name ```run.py``` under the main folder, the DLO merging & Interpolation functionality is implemented in ```core.py```, color filtering object is added in a seperate file ```colorFilter.py```.

### Reference
DeepLabV3+ implementation based on [https://github.com/VainF/DeepLabV3Plus-Pytorch](https://github.com/VainF/DeepLabV3Plus-Pytorch)

The origin source code from [https://github.com/lar-unibo/fastdlo](https://github.com/lar-unibo/fastdlo)

