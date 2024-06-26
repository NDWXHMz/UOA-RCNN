﻿# UOA-RCNN : detecting anything with Unknown Object Aware RCNN





# Abstract

<div>Unknown Object Detection has received increasing attention due to its adaptability to open scenarios in the real world. Previous methods confused unknown objects with non-objects and made unreasonable selections for unknown predictions, leading to inaccurate unknown detection. Inspired from the known object detection, we propose a novel unknown object detection method, named Unknown Object Aware RCNN (UOA-RCNN), which addresses the above issues. Firstly, the Unknown Object Aware Module is proposed, which learns Universal Objectness Score (UOS) from known objects that can be generalized to unknown objects, significantly improving the discriminability between objects and non-objects. Subsequently, the *Known Object Probability* is introduced to optimize the identified unknown objects, further suppressing potential non objects. Finally, we design an unknown object mining scheme based on the UOS that can accurately locate unknown objects and remove redundant results during prediction. Experimental results show that our method achieves the best performance on the unknown object detection benchmark. </div>



# Requirements
```bash
pip install -r requirements.txt
```

In addition, install detectron2 following [here](https://detectron2.readthedocs.io/en/latest/tutorials/install.html).

# Dataset Preparation

The datasets can be downloaded using this [link](https://drive.google.com/drive/folders/1Mh4xseUq8jJP129uqCvG9cSLdjqdl0Jo?usp=sharing).

**PASCAL VOC**

Please put the corresponding json files in Google Cloud Disk into ./anntoations

Please download the JPEGImages data from the [link](https://drive.google.com/file/d/1n9C4CiBURMSCZy2LStBQTzR17rD_a67e/view?usp=sharing) provided by [VOS](https://github.com/deeplearning-wisc/vos).

The VOC dataset folder should have the following structure:
<br>

     └── VOC_DATASET_ROOT
         |
         ├── JPEGImages
         ├── voc0712_train_all.json
         ├── voc0712_train_completely_annotation200.json
         └── val_coco_format.json

**COCO**

Please put the corresponding json files in Google Cloud Disk into ./anntoations

The COCO dataset folder should have the following structure:
<br>

     └── COCO_DATASET_ROOT
         |
         ├── annotations
            ├── xxx (the original json files)
            ├── instances_val2017_coco_ood.json
            ├── instances_val2017_mixed_ID.json
            └── instances_val2017_mixed_OOD.json
         ├── train2017
         └── val2017

# Training
```bash
python train_net.py --dataset-dir VOC_DATASET_ROOT --num-gpus 2 --config-file VOC-Detection/faster-rcnn/UOARCNN.yaml --random-seed 0 --resume
```


# Pretesting
The function of this process is to obtain the threshold, which only uses part of the training data.

pretrained model: TODO

```bash
sh pretest.sh
```

# Evaluation on the VOC



```bash
python apply_net.py --dataset-dir VOC_DATASET_ROOT --test-dataset voc_custom_val  --config-file VOC-Detection/faster-rcnn/UOARCNN.yaml --inference-config Inference/standard_nms.yaml --random-seed 0 --image-corruption-level 0 --visualize 0
```

# Evaluation on the COCO-OOD
```bash
sh test_ood.sh
```

# Evaluation on the COCO-Mix

```bash
sh test_mixed.sh
```

# Visualize prediction results
```bash
sh vis.sh
```

