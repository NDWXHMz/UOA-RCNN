import json
import os, cv2
import json
import shutil
import cv2

def select(json_path, outpath, image_path):
    json_file = open(json_path)
    infos = json.load(json_file)
    images = infos["images"]
    annos = infos["annotations"]
    assert len(images) == len(images)
    for i in range(len(images)):
        im_id = images[i]["id"]
        im_path = image_path + "/" + images[i]["file_name"]
        img = cv2.imread(im_path)
        for j in range(len(annos)):
            if annos[j]["image_id"] == im_id:
                x, y, w, h = annos[j]["bbox"]
                x, y, w, h = int(x), int(y), int(w), int(h)
                x2, y2 = x + w, y + h
                # object_name = annos[j][""]
                img = cv2.rectangle(img, (x, y), (x2, y2), (255, 0, 0), thickness=2)
                img_name = outpath + "/" + images[i]["file_name"]
                cv2.imwrite(img_name, img)
                # continue
        print(i)

if __name__ == "__main__":

  
    train_json = r'/data/datasets/Detection/coco/annotations/instances_val2017_mixed_OOD.json'
    train_path = r'/data/datasets/Detection/coco/val2017'
    visual_output = r'/data/lhm/UnSniffer_2/detection/coco_vis_output'

    


    select(train_json, visual_output, train_path)

