## MRSN
This is Official Repository for "Mutually Reinforcing Structure with Proposal Contrastive Consistency for Few-Shot Object Detection", ECCV 2022.
![Framework of our method](https://github.com/MMatx/MRSN/blob/main/Framework.png)
## Training
* 1. Training a plain detector.
* 2. Using the balanced dataset to fine-tune the classifier and regressor of the detector.
* 2. Executing UD-CutMix by recombining labeled novel instances with base images to construct a new synthetic set.
* 3. Training Mutually Reinforcing Structure Network using base dataset and synthetic set.

## Acknowledgement
This repo is developed based on [TFA](https://github.com/ucbdrive/few-shot-object-detection) , [Detectron2](https://github.com/facebookresearch/detectron2) , [unbiased-teacher](https://github.com/facebookresearch/unbiased-teacher). 
