python apply_net.py --dataset-dir /data/lhm/datasets/coco --test-dataset coco_mixed_val --config-file VOC-Detection/faster-rcnn/UnSniffer.yaml --inference-config Inference/standard_nms.yaml --random-seed 0 --image-corruption-level 0 --visualize 0

cd evaluator/

python eval.py --dataset-dir /data/lhm/datasets/coco --test-dataset coco_mixed_val --outputdir ../output/  --config-file VOC-Detection/faster-rcnn/UnSniffer.yaml --inference-config Inference/standard_nms.yaml --random-seed 0 --image-corruption-level 0

# python aose.py --dataset-dir /data/datasets/Detection/coco --test-dataset coco_mixed_val --outputdir ../output/  --config-file VOC-Detection/faster-rcnn/UnSniffer.yaml --inference-config Inference/standard_nms.yaml --random-seed 0 --image-corruption-level 0

# python WI.py --dataset-dir /data/datasets/Detection/coco --test-dataset coco_mixed_val --outputdir ../output/  --config-file VOC-Detection/faster-rcnn/UnSniffer.yaml --inference-config Inference/standard_nms.yaml --random-seed 0 --image-corruption-level 0

# cd ..   