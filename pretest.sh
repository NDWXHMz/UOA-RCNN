python apply_net.py --dataset-dir VOC_DATASET_ROOT --test-dataset voc_completely_annotation_pretest --config-file VOC-Detection/faster-rcnn/UnSniffer.yaml --inference-config Inference/standard_nms.yaml --random-seed 0 --image-corruption-level 0 --visualize 0  --pretest True

cd evaluator/

python grid_traverse.py --dataset-dir VOC_DATASET_ROOT --test-dataset voc_completely_annotation_pretest --outputdir ../output/  --config-file VOC-Detection/faster-rcnn/UnSniffer.yaml --inference-config Inference/standard_nms.yaml --random-seed 0 --image-corruption-level 0

python energy_thresh.py  --dataset-dir VOC_DATASET_ROOT --config-file VOC-Detection/faster-rcnn/UnSniffer.yaml --inference-config Inference/standard_nms.yaml --image-corruption-level 0

cd ..