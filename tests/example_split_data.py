
#%%
from image_dataspliter.image_dataspliter import split_data
from image_dataspliter.feat import ImgPropertySetReturnType
import os
from glob import glob
import json
import shutil
import random
#%%
#imgdir = "/home/lin/codebase/experiment_for_image_dataspliter/mixed_dataset"
#coco_annpath = "/home/lin/codebase/experiment_for_image_dataspliter/mixed_dataset_combined_annotations.json"


imgdir = "/home/lin/codebase/experiment_for_image_dataspliter/mini_mixed_dataset"
coco_annpath = "/home/lin/codebase/experiment_for_image_dataspliter/mini_mixed_dataset_ann.json"

# img_paths = glob(f"{imgdir}/*")
# img_names = [os.path.basename(img_path) for img_path in img_paths]

# selected_img = random.sample(img_paths, 3000)

# img_names = [os.path.basename(img_path) for img_path in selected_img]


#%%
# total_num_imgs = len(img_names)

# img_property_set = ImgPropertySetReturnType(img_names=img_names, 
#                                             img_paths=img_paths, 
#                                             total_num_imgs=total_num_imgs, 
#                                             max_num_clusters=None
#                                             )

obj_img_property_set = ImgPropertySetReturnType(img_names=None, 
                                                img_paths=None, 
                                                total_num_imgs=None, 
                                                max_num_clusters=None
                                                )



    
#%%

# import pandas as pd

# insitu_df =pd.read_csv("/home/lin/codebase/image_dataspliter/tests/insitu_obj_based_data_split.csv")


# insitu_df.clusters.count() #value_counts()

# noobj_imgs = ["20221205_151635_jpg.rf.fc1a6a7e66902bf0246bb3247bb8553b.jpg",
#  "20221205_151610_jpg.rf.55dd0fd953b0b015aebd24eafb2e9f50.jpg",
# "20221205_151638_jpg.rf.e4e11feb6dbe02c8ed5a37231e091ec9.jpg",
# "20221205_151640_jpg.rf.fed72007df790f69a7d70feb71325bb2.jpg",
# "20221205_151642_jpg.rf.e9d45653d8fd88f65023a5ff78c66107.jpg",
# "20221205_151734_jpg.rf.53a9f0f441818613483c2ab551749842.jpg",
# "20221205_151742_jpg.rf.03a1cc3bbef021be4058625c6ae30bb3.jpg",
# "20221205_151757_jpg.rf.abb814d9775f827ef0bc586314e11fe7.jpg",
# "20221205_151759_jpg.rf.627ed0e3f7956514c03575151d335d4c.jpg",
# "20221205_161605_jpg.rf.30217721f2cc3b7f52f112561fc42273.jpg",
# "20221205_161611_jpg.rf.bb63d4025ef646d8630d356a29c14c5f.jpg",
# "20221206_101346_jpg.rf.5bce3ea607741ab46813e716e5603654.jpg",
# "20221206_104631_jpg.rf.90dc889d13e8af72b44347707edfe1da.jpg"
# ]

# #%%
# insitu_df[insitu_df.image_names.isin(noobj_imgs)].clusters.unique()
# # %%
# with open(coco_annpath, 'r') as f:
#     cocodata = json.load(f)
#     print(len(cocodata["images"]))
# %%

### same images being split into different imgae objects
### despite insitu = True
["/home/lin/codebase/image_dataspliter/tests/crop_objs/20221227_150127_jpg.rf.5e41f91fed5ffeee975121889f8b62ed_0.png"
    "/home/lin/codebase/image_dataspliter/tests/crop_objs/20221227_150127_jpg.rf_1.png"
    "/home/lin/codebase/image_dataspliter/tests/crop_objs/20221227_150127_jpg_2.png",
    
]


["/home/lin/codebase/image_dataspliter/tests/crop_objs/20221227_150157_jpg_2.png",
 "/home/lin/codebase/image_dataspliter/tests/crop_objs/20221227_150157_jpg_3.png",
 "/home/lin/codebase/image_dataspliter/tests/crop_objs/20221227_150157_jpg_4.png"
]


#%%
def subset_coco_annotations(img_list, coco_annotation_file,
                            save_annotation_as
                            ):
    with open(coco_annotation_file, "r") as f:
        coco_data = json.load(f)
    name_to_id = {os.path.basename(img["file_name"]): img["id"] for img in coco_data["images"]}
    image_ids = [name_to_id[os.path.basename(imgname)] for imgname in img_list if os.path.basename(imgname) in name_to_id]
    
    annotations = []
    for ann in coco_data["annotations"]:
        if ann["image_id"] in image_ids:
            annotations.append(ann)
    images = []
    for image in coco_data["images"]:
        if image["id"] in image_ids:
            images.append(image)
    
    subset_coco_annotations = {}       
    for field in coco_data:
        if field not in ["annotations", "images"]:
            subset_coco_annotations[field] = coco_data[field]
    subset_coco_annotations["annotations"] = annotations
    subset_coco_annotations["images"] = images
    ann_dir = os.path.dirname(save_annotation_as)
    
    if not os.path.exists(ann_dir):
        os.makedirs(ann_dir)
    with open(save_annotation_as, "w") as f:
        json.dump(subset_coco_annotations, f)
    return subset_coco_annotations
        

def subset_data(img_dir, img_list, save_img_dir, 
                coco_annotation_file,
                save_annotation_as
                ):
    os.makedirs(save_img_dir, exist_ok=True)
    for img in img_list:
        img = os.path.basename(img)
        shutil.copy(os.path.join(img_dir, img), save_img_dir)
    subseted_coco_ann = subset_coco_annotations(img_list=img_list,
                                                coco_annotation_file=coco_annotation_file,
                                                save_annotation_as=save_annotation_as
                                                )
    return subseted_coco_ann


#%%
# save_img_dir = "/home/lin/codebase/experiment_for_image_dataspliter/mini_mixed_dataset"
# save_annotation_as = "/home/lin/codebase/experiment_for_image_dataspliter/mini_mixed_dataset_ann.json"
# subset_data(img_dir=imgdir, img_list=selected_img,
#             coco_annotation_file=coco_annpath,
#             save_img_dir=save_img_dir,
#             save_annotation_as=save_annotation_as
#             )




#%%
# insitu_imgsplit = "/home/lin/codebase/image_dataspliter/tests/full_image_data_split.json"

# with open(insitu_imgsplit, "r") as f:
#     insitu_split_results = json.load(f)
    
# train_set = insitu_split_results["train_set"]
# val_set = insitu_split_results["val_set"]   
# save_img_dir = "/home/lin/codebase/experiment_for_image_dataspliter/full_image_cocoa_ripeness_instv2i/images/val"
# save_annotation_as = "/home/lin/codebase/experiment_for_image_dataspliter/full_image_cocoa_ripeness_instv2i/annotations/instances_val.json"
# subset_data(img_dir=imgdir, img_list=val_set,
#             coco_annotation_file=coco_annpath,
#             save_img_dir=save_img_dir,
#             save_annotation_as=save_annotation_as
#             )
# %%
#/home/lin/codebase/otx_env/lib/python3.10/site-packages/otx/recipe/instance_segmentation/maskrcnn_r50.yaml

#%%

import tensorflow as tf
# %%
tf.test.is_built_with_cuda()
# %%
tf.test.is_gpu_available()
# %%
tf.config.list_physical_devices("GPU")
# %%
#%%
# from glob import glob
# imgpaths = glob("/home/lin/codebase/pipeapple_instance_seg/5_classes_pineapple2.v1i.coco-segmentation/valid/*")
# for img in imgpaths:
#     #print(f"before removal: {img}")
#     if os.path.splitext(img)[-1] == ".Identifier":
#         print(img)
#         os.remove(img)



if __name__ == "__main__":
    split_results = split_data(img_property_set=img_property_set,
                                use_object_features=False, 
                                parallelize=False, 
                                insitu=False,
                                include_testsplit=True,
                                coco_annotation_file=coco_annpath,
                                img_dir=imgdir
                                )

    with open("/home/lin/codebase/image_dataspliter/tests/mixed_dataset/fullimage_data_split.json", "w") as f:
        json.dump(split_results, f)
    
    
    # Processed 14530 out of 34694
# Getting object features - non insitu:  42%|█████████████████▏                       | 14530/34694 [8:08:50<23:16:08,  4.15s/it]
##### reduce images for spliting to about 5,000 so it can fit into memory 


#%%
crop_dir = "/home/lin/codebase/image_dataspliter/tests/crop_objs"

len(glob(f"{crop_dir}/*"))
# %%
