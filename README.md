
<div align="center">
<h2> FASTDLO: Fast Deformable Linear Objects Instance Segmentation </h2>

 :page_with_curl:  [IEEE Xplore](https://ieeexplore.ieee.org/document/9830852)  :page_with_curl:	
</div>

### Notation 

the semantic segmentation module is inherited from above paper mentioned algorithm, we modified this as it can only detect only one single DLO with detection path, which can be used in the tracking algorithm e.g. CPD.

### Installation

Main dependencies:

The program successfully run on a Laptop with Ubuntu 22.04 LTS, with 32 GB RAM and Nvidia RTX 1650 mobile graphic card
```
python (3.8)
pytorch (1.4.0)
opencv 
pillow 
scikit-image 
scipy 
shapely 
```

Installation (from inside the main project directory):
```
sudo pip install .
```

### Models' weights

Download the [weights](https://drive.google.com/file/d/1OVcro53E_8oJxRPHqGy619rBNoCD3rzT/view?usp=sharing) and place them inside a ```weights``` folder.


### Usage

import as a standard python package with ```from fastdlo.core import Pipeline```.

Then initialize the class ``` p = Pipeline(checkpoint_siamese_network, checkpoint_segmentation_network) ```

the inference can be obtained with ```pred = p.run(source_img) ```.

a example run file is prepared with name ```intelRealSense_run.py``` under the main folder


### Reference
DeepLabV3+ implementation based on [https://github.com/VainF/DeepLabV3Plus-Pytorch](https://github.com/VainF/DeepLabV3Plus-Pytorch)
The origin source code from [https://github.com/lar-unibo/fastdlo](https://github.com/lar-unibo/fastdlo)

