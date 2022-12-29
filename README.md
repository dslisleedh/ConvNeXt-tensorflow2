# ConvNeXt-tensorflow2
Unofficial implementation of ConvNeXt using tensorflow2 and keras. 
paper: https://arxiv.org/abs/2201.03545

You can run train example code by `train.py`

    conda env create -f environment.yaml
    conda activate convnext
    python train.py model=$model_name  ex)convnext_b
    
Implemented models:  
 - convnext_t
 - convnext_s
 - convnext_b
 - convnext_l
 - convnext_xl
 - convnext_es (Extremely small size for personal use. It is not described in the paper.)
