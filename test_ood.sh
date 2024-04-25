CUDA_VISIBLE_DEVICES=1 python apply_net.py --dataset-dir /data/lhm/datasets/coco --test-dataset coco_extended_ood_val --config-file VOC-Detection/faster-rcnn/UnSniffer.yaml --inference-config Inference/standard_nms.yaml --random-seed 0 --image-corruption-level 0 --visualize 0

cd evaluator/

python eval.py --dataset-dir /data/lhm/datasets/coco --test-dataset coco_extended_ood_val --outputdir ../output/  --config-file VOC-Detection/faster-rcnn/UnSniffer.yaml --inference-config Inference/standard_nms.yaml --random-seed 0 --image-corruption-level 0

cd ..
