#%%
import pandas as pd
import os
from glob import glob
import shutil
from example_split_data import subset_coco_annotations

def export_data(data_path, img_dir, coco_annotation_path, save_dir):
    
    os.makedirs(save_dir, exist_ok=True)
    data = pd.read_csv(data_path)
    split_type = data.split_type.unique().tolist()
    imgs = glob(f"{img_dir}/*")
    for split in split_type:
        split_imglist = data[data.split_type==split].image_names.tolist()
        split_imgpath = [img for img in imgs if os.path.basename(img) in split_imglist]
        save_img_dir = os.path.join(save_dir, "images", split)
        save_ann_dir = os.path.join(save_dir, "annotations", split)
        os.makedirs(save_img_dir, exist_ok=True)
        os.makedirs(save_ann_dir, exist_ok=True)
        
        if split_imglist:
            for img in split_imgpath:
                shutil.copy(img, save_img_dir)
            save_annotation_as = os.path.join(save_ann_dir, f"instances_{split}.json" )
            
            splittype_cocodata = subset_coco_annotations(img_list=split_imglist, 
                                                        coco_annotation_file=coco_annotation_path,
                                                        save_annotation_as=save_annotation_as
                                                        )
        else:
            print(f"split_type: {split} has no images")
            

if __name__ == "__main__":
    non_insitu_path = "/home/lin/codebase/image_dataspliter/tests/fullimage_data_split.csv"
    img_dir = "/home/lin/codebase/experiment_for_image_dataspliter/mini_mixed_dataset"
    coco_annotation_file = "/home/lin/codebase/experiment_for_image_dataspliter/mini_mixed_dataset_ann.json"
    save_dir = "/home/lin/codebase/experiment_for_image_dataspliter/fullimage_mini_mixed_dataset"
    
    #noninsitu_df = pd.read_csv(non_insitu_path)
    export_data(data_path=non_insitu_path, 
                img_dir=img_dir,
                coco_annotation_path=coco_annotation_file,
                save_dir=save_dir
                )