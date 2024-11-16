import os
import cv2
import numpy as np
from typing import List, Dict
from pycocotools.coco import COCO
import cv2
import numpy as np
from PIL import Image
from clusteval import clusteval
import pandas as pd
import json
from .feat import get_object_features, get_obj_features_per_img_non_insitu_wrapper
from .feat import get_obj_features_per_img_non_insitu, img_feature_extraction_implementor
from .feat import ImgPropertySetReturnType, run_multiprocess
import multiprocessing
from copy import deepcopy
from tqdm import tqdm

def get_objects(imgname, coco, img_dir):
    try:
        val = next(obj for obj in coco.imgs.values() if obj["file_name"] == imgname)
    except StopIteration:
        raise ValueError(f"Image {imgname} not found in COCO dataset.")
    
    img_id = val['id']
    img_info = coco.loadImgs(img_id)[0]
    img_path = os.path.join(img_dir, imgname)
    image = cv2.imread(img_path)

    # Get annotation IDs for the image
    ann_ids = coco.getAnnIds(imgIds=img_id)
    anns = coco.loadAnns(ann_ids)
    img_obj = []

    for ann in anns:
        mask = coco.annToMask(ann)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            cropped_object = image[y:y+h, x:x+w]

            # Apply the mask to the cropped object
            mask_cropped = mask[y:y+h, x:x+w]
            #print(f"mask_cropped: {mask_cropped.shape}")
            cropped_object = cv2.bitwise_and(cropped_object, cropped_object, mask=mask_cropped)
            
            # Remove the background (set to transparent)
            cropped_object = cv2.cvtColor(cropped_object, cv2.COLOR_BGR2RGBA)
            cropped_object[:, :, 3] = mask_cropped * 255

            img_obj.append(cropped_object)
    
    os.makedirs(name="crop_objs", exist_ok=True)
    for img_count, each_img_obj in enumerate(img_obj):
        imgname = os.path.splitext(imgname)[0]
        cv2.imwrite(filename=f"crop_objs/{imgname}_{img_count}.png", img=each_img_obj)
    
    return img_obj


def get_objects_keep_imgdim(imgname, coco, img_dir, save_crop_objs_dir="crop_objs") -> List:
    try:
        val = next(obj for obj in coco.imgs.values() if obj["file_name"] == imgname)
    except StopIteration:
        raise ValueError(f"Image {imgname} not found in COCO dataset.")
    
    img_id = val['id']
    img_info = coco.loadImgs(img_id)[0]
    img_path = os.path.join(img_dir, imgname)
    image = cv2.imread(img_path)
    #print(f"img_path: {img_path}")
    #print(f"image: {image}")

    # Get annotation IDs for the image
    ann_ids = coco.getAnnIds(imgIds=img_id)
    anns = coco.loadAnns(ann_ids)
    #print(f"len of anns: {len(anns)}")
    #print(f"anns:  {anns}")
    img_obj = []

    for ann in anns:
        mask = coco.annToMask(ann)

        # Apply the mask to the image to get the segmented object
        segmented_object = cv2.bitwise_and(image, image, mask=mask)
        
        # Remove the background (set to transparent)
        segmented_object = cv2.cvtColor(segmented_object, cv2.COLOR_BGR2BGRA)
        segmented_object[:, :, 3] = mask * 255

        img_obj.append(segmented_object)
    if len(img_obj) != 0:
        default_img = np.zeros_like(img_obj[0])
        for obj in img_obj:
            img_obj = cv2.add(default_img, obj, default_img)
    elif len(img_obj) == 0:
        img_obj = np.zeros_like(image)
        img_obj = cv2.cvtColor(img_obj, cv2.COLOR_BGR2BGRA)
        #print(f"{imgname} has no object hence an empty images is created")
    if not isinstance(img_obj, list):
        img_obj = [img_obj]
    os.makedirs(name=save_crop_objs_dir, exist_ok=True)
    for img_count, each_img_obj in enumerate(img_obj):
        imgname = os.path.splitext(imgname)[0]
        cv2.imwrite(filename=f"{save_crop_objs_dir}/{imgname}_{img_count}.png", img=each_img_obj)
    
    return img_obj


def get_objects_per_img(coco_annotation_file, img_dir, coco=None, img_names=None):
    if not coco:
        coco = COCO(annotation_file=coco_annotation_file)
    if not img_names:
        img_names = [obj["file_name"] for obj in coco.imgs.values()]
    if not isinstance(img_names, list):
        img_names = [img_names]
    img_objects = {}
    total = len(img_names)
    for idx, imgname in tqdm(enumerate(img_names), total=total,
                                       desc="Getting objects kepts in positions"
                                       
                            ):
        img_objs = get_objects_keep_imgdim(imgname, coco, img_dir)
        if img_objs:
            img_objs = [cv2.cvtColor(obj, cv2.COLOR_RGBA2RGB) for obj in img_objs]
            img_objects[imgname] = img_objs 
            print(f"Processed {idx+1} out of {total}")       
    return img_objects

def get_objects_per_img_wrapper(args):
    img_objects = get_objects_per_img(**args)
    return img_objects
    
def get_obj_features_per_img_insitu(img_objects,img_resize_width,
                            img_resize_height,
                            model_family, model_name,
                            img_normalization_weight,
                            seed,
                            img_property_set: ImgPropertySetReturnType
                            ):
    img_feature = {}
    total = len(img_objects)
    for idx, img_object in tqdm(enumerate(img_objects.items()), total=total,
                                 desc="Insitu: Getting object features"
                                 ):
        imgname, objs = img_object[0], img_object[1]
        #for imgname, objs in img_object:
        feature = get_object_features(obj_imgs=objs, 
                                    img_resize_width=img_resize_width,
                                    img_resize_height=img_resize_height,
                                    model_family=model_family, model_name=model_name,
                                    img_normalization_weight=img_normalization_weight,
                                    seed=seed,
                                    )
        img_feature[imgname] = feature
        print(f"Processed {idx+1} out of {total}")
    img_property_set.img_names = [img_name for img_name in img_feature.keys()]
    img_property_set.features = [feat for feat in img_feature.values()]
    return img_property_set

def get_obj_features_per_img_insitu_wrapper(args):
    img_property_set = get_obj_features_per_img_insitu(**args)
    return img_property_set

def cluster_img_features(img_property_set: ImgPropertySetReturnType) -> pd.DataFrame:
    img_names = img_property_set.img_names
    img_feats = img_property_set.features
    featarray = np.array(img_feats)
    ce = clusteval()
    results = ce.fit(featarray)
    clusters = results["labx"]
    imgcluster_dict = {"image_names": img_names, "clusters": clusters}
    imgclust_df = pd.DataFrame.from_dict(imgcluster_dict)
    imgclust_df.to_csv("cluster_df.csv")
    return imgclust_df
        

def object_based_cluster_images_insitu(coco_annotation_file, img_dir,
                                             img_property_set: ImgPropertySetReturnType,
                                             seed=2024, img_resize_width=224,
                                            img_resize_height=224,
                                            model_family="efficientnet",
                                            model_name="EfficientNetB0",
                                            img_normalization_weight="imagenet",
                                            
                                            ):
    
    img_objects = get_objects_per_img(coco_annotation_file=coco_annotation_file,
                                        img_dir=img_dir
                                        )
    #print(f"number of img objects: {len(img_objects)}")
    img_property_set = get_obj_features_per_img_insitu(img_objects=img_objects, 
                                           img_resize_width=img_resize_width,
                                            img_resize_height=img_resize_height,
                                            model_family=model_family,
                                            model_name=model_name,
                                            img_normalization_weight=img_normalization_weight,
                                            seed=seed, 
                                            img_property_set=img_property_set
                                            )  
    cluster_df = cluster_img_features(img_property_set=img_property_set) 
    return cluster_df


def object_based_cluster_images_non_insitu(img_dir, coco_annotation_file,
                                        img_property_set
                                        ):
    img_property_set= get_obj_features_per_img_non_insitu(coco_annotation_file, img_dir=img_dir,
                                                        img_property_set=img_property_set
                                                        )
    cluster_df = cluster_img_features(img_property_set=img_property_set) 
    return cluster_df

def cluster_with_full_image(img_property_set):
    img_property_set = img_feature_extraction_implementor(img_property_set)
    cluster_df = cluster_img_features(img_property_set=img_property_set) 
    return cluster_df

def clusters_with_full_image_multiprocess(img_property_set, **kwargs):
    img_property_set = run_multiprocess(img_property_set)
    cluster_df = cluster_img_features(img_property_set=img_property_set) 
    return cluster_df

def object_based_cluster_images_insitu_multiprocess(coco_annotation_file, img_dir,
                                                          img_property_set,
                                                          seed=2024, img_resize_width=224,
                                                        img_resize_height=224,
                                                        model_family="efficientnet",
                                                        model_name="EfficientNetB0",
                                                        img_normalization_weight="imagenet"
                                                        ):
    coco = COCO(coco_annotation_file)
    img_names = [obj["file_name"] for obj in coco.imgs.values()]
    args_objects_per_img = [{"coco_annotation_file": coco_annotation_file,
                            "img_dir": img_dir, 
                            "coco": coco,
                            "img_names": img_name
                            } for img_name in img_names
                            ]
    objects_results = parallelize_func(args=args_objects_per_img, func=get_objects_per_img_wrapper)
    print("multiprocess of get_objects_per_img completed")
    args_get_obj_features_per_img = [{"img_objects": res, "img_resize_width": img_resize_width,
                                        "img_resize_height": img_resize_height, 
                                        "model_family": model_family,
                                        "model_name":model_name, 
                                        "img_normalization_weight": img_normalization_weight,
                                        "seed": seed, "img_property_set": img_property_set
                                        } for res in objects_results
                                    ]
    feat_results = parallelize_func(args=args_get_obj_features_per_img, 
                                    func=get_obj_features_per_img_insitu_wrapper
                                    )
    print("Completed multiprocesssing of get_obj_features_per_img_insitu")
    img_names, features = [], []
    for res in feat_results:
        img_names.extend(res.img_names)
        features.extend(res.features)
    img_property_set.img_names = img_names
    img_property_set.features = features
    
    print(f"Started clustering")
    cluster_df = cluster_img_features(img_property_set=img_property_set) 
    return cluster_df
    
def object_based_cluster_images_non_insitu_multiprocess(img_dir, coco_annotation_file,
                                                    img_property_set
                                                    ):
    coco = COCO(coco_annotation_file)
    img_names = [obj["file_name"] for obj in coco.imgs.values()]
    args_objects = [{"coco_annotation_filepath": coco_annotation_file,
                     "coco": deepcopy(coco),
                    "img_dir": img_dir,
                    "img_property_set": img_property_set,
                    "img_names": img_name
                    } for img_name in img_names
                    ]
    img_names, features = [], []
    img_property_set_results = parallelize_func(args=args_objects, 
                                                func=get_obj_features_per_img_non_insitu_wrapper
                                                )
    print(f"Completed multiprocessing of get_obj_features_per_img_non_insitu")
    for res in img_property_set_results:
        img_names.extend(res.img_names)
        features.extend(res.features)
    img_property_set.img_names = img_names
    img_property_set.features = features
    
    print("Started clustering")
    cluster_df = cluster_img_features(img_property_set=img_property_set) 
    return cluster_df

def parallelize_func(args, func):
    chunksize = max(1, len(args) // 10)
    num_processes = multiprocessing.cpu_count()
    from tqdm import tqdm
    with multiprocessing.Pool(num_processes) as p:
        results = list(
                    tqdm(
                        p.imap_unordered(
                            func, args, chunksize=chunksize
                        ),
                        total=len(args),
                    )
                )
    return results