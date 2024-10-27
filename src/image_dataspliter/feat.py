
#%%
from typing import NamedTuple, Union, Tuple, List
import tensorflow as tf
import torch
from PIL import Image
import random
import numpy as np
import multiprocessing
from dataclasses import dataclass
from typing import List
from glob import glob
import os
from multiprocessing import Process, Lock
import cv2
from pycocotools.coco import COCO
from tensorflow.keras.layers import Add
from clusteval import clusteval
import pandas as pd
#%%
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

tf.config.set_visible_devices([], 'GPU')

class ModelPreprocessReturn(NamedTuple):
    model: str
    preprocess: str
    
@dataclass
class ImgPropertySetReturnType:
    img_names: List
    img_paths: List
    total_num_imgs: int
    max_num_clusters: int    
    
def load_model_and_preprocess(input_shape, model_family, model_name, weight
                              ):
    model = getattr(tf.keras.applications, model_name)(input_shape=input_shape,
                                                      weights=weight,
                                                      include_top=False)
    preprocess = getattr(tf.keras.applications, model_family).preprocess_input
    model_preprocess_result = ModelPreprocessReturn(model=model, preprocess=preprocess)
    return model_preprocess_result

class FeatureExtractor(object):
    def __init__(self, seed=2024, img_resize_width=224,
                 img_resize_height=224, model_family="efficient",
                 model_name="EfficientNetB0",
                 img_normalization_weight="imagenet"
                 ):
        self.seed=seed
        self.img_resize_width=img_resize_width
        self.img_resize_height=img_resize_height
        self.model_family=model_family
        self.model_name=model_name
        self.image_shape=(img_resize_height, img_resize_width, 3)
        self.img_normalization_weight=img_normalization_weight
    import numpy as np
    import tensorflow as tf    
    import random
    def set_seed_consistently(self, seed=2024):
        if seed:
            self.seed=seed
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        
    def load_and_resize_image(self, img_path, width=None, height=None):
        if not width:
            width = self.img_resize_width
        
        if not height:
            height = self.img_resize_height
            
        img = Image.open(img_path).resize((width, height))
        return img
    
    def load_model_and_preprocess_func(self, input_shape=None,
                                       model_family=None, model_name=None,
                                       weight=None
                                       ):
        if not input_shape:
            input_shape = self.image_shape
            
        if not model_family:
            model_family = self.model_family
        if not model_name:
            model_name = self.model_name
        if not weight:
            weight = self.img_normalization_weight
        
        model = getattr(tf.keras.applications, model_name)(input_shape=input_shape,
                                                      weights=weight,
                                                      include_top=False)
        preprocess = getattr(tf.keras.applications, model_family).preprocess_input
        self.model_preprocess_result = ModelPreprocessReturn(model=model, preprocess=preprocess)
        return self.model_preprocess_result
    
    def _check_model_preprocess_exist(self):
        if hasattr(self, "model_preprocess_result"):
            model, preprocess = self.model_preprocess_result
            return model, preprocess
        else:
            model, preprocess = self.load_model_and_preprocess_func()
            return model, preprocess
        
    def get_feature_extractor(self, model=None):
        if not model:
            model, preprocess = self._check_model_preprocess_exist()
            
        self.inputs = model.inputs
        x = model(self.inputs)
        outputs = tf.keras.layers.GlobalAveragePooling2D()(x)
        self.feature_extractor = tf.keras.Model(inputs=self.inputs, outputs=outputs,
                                                name="feature_extractor")
        return self.feature_extractor
    
    def extract_features(self, inputs, model=None, preprocess=None):
        if not model or not preprocess:
            model, preprocess = self._check_model_preprocess_exist()
        x = preprocess(inputs)
        preds = model(x)
        return preds[0]
    
    def load_image_for_inference(self, image_path, img_shape=None):
        if not img_shape:
            img_shape = self.image_shape
        image = tf.io.read_file(image_path)
        x = tf.image.decode_image(image, channels=img_shape[2])
        x = tf.image.resize(x, (img_shape[0], img_shape[1]))
        x = tf.expand_dims(x, axis=0)
        return x
    
    def get_images_features(self, img_property_set, feature_extractor=None,
                            preprocess=None,
                            use_cropped_imgs=False,
                            use_merged_cropped_imgs=False
                            ):
        if not feature_extractor:
            if hasattr(self, "feature_extractor"):
                feature_extractor = self.feature_extractor
            else:
                feature_extractor = self.get_feature_extractor()
                
        if not preprocess:
            model, preprocess = self._check_model_preprocess_exist()
            
        images = []
        features = []
        
        if use_cropped_imgs and use_merged_cropped_imgs:
            raise Exception("both not provided")
        
        if use_merged_cropped_imgs:
            img_paths = sorted(img_property_set.merged_cropped_img_paths)
        if use_cropped_imgs:
            img_paths = sorted(img_property_set.cropped_img_paths)
        else:
            img_paths = sorted(img_property_set.img_paths)
            
        if use_cropped_imgs:
            img_property_set.cropped_imgs = images
            img_property_set.cropped_img_features = features
            
        img_property_set.imgs = images
        img_property_set.features = features
        
        
def get_objects(imgname, coco, img_dir):
    val = [obj for obj in coco.imgs.values() if obj["file_name"] == imgname][0]
    img_id = val['id']
    img_info = coco.loadImgs(img_id)[0]
    img_path = os.path.join(img_dir, imgname)
    image = cv2.imread(img_path)

    # Get annotation IDs for the image
    ann_ids = coco.getAnnIds(imgIds=img_id)
    anns = coco.loadAnns(ann_ids)
    img_obj = []
    for ann in anns:
        segmentation = ann['segmentation']
        mask = coco.annToMask(ann)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            cropped_object = image[y:y+h, x:x+w]
            img_obj.append(cropped_object)
    return img_obj


def get_object_features(obj_imgs, 
                        img_resize_width,
                        img_resize_height,
                        model_family, model_name,
                        img_normalization_weight,
                        seed, #images_list, features_list, 
                        #model_artefacts_dict, #lock
                        ):
    from tensorflow.keras.layers import Add
    import tensorflow as tf
    feat_extract = FeatureExtractor(seed=seed, img_resize_width=img_resize_width,
                                    img_resize_height=img_resize_height, 
                                    model_family=model_family,
                                    model_name=model_name,
                                    img_normalization_weight=img_normalization_weight
                                    )
    feature_list = []
    for obj_img in obj_imgs:
        channels = obj_img.shape[2]
        #Image.fromarray(obj_img).show()
        img_tensor = tf.convert_to_tensor(obj_img)
        img_resized = tf.image.resize_with_pad(image=img_tensor, target_height=img_resize_height, target_width=img_resize_width)
        img_for_infer = tf.expand_dims(img_resized, axis=0)
        feat_extract.set_seed_consistently()
        model, preprocess = feat_extract.load_model_and_preprocess_func()
        feature_extractor = feat_extract.get_feature_extractor(model)
        feature = feat_extract.extract_features(img_for_infer, feature_extractor, preprocess)
        feature_list.append(feature)
    
    if len(feature_list) > 1:
        print(f"len(feature_list): {len(feature_list)}")
        feature = Add()(feature_list)#/len(feature_list)
    else:
        feature = feature_list[0]
    return feature


def get_imgs_and_extract_features(img_path, img_resize_width,
                                img_resize_height,
                                model_family, model_name,
                                img_normalization_weight,
                                seed, return_img_path=False,
                                #use_objects_in_image_as_features=False,
                                objects_insitu=False,
                                coco_annotation_filepath=None#images_list, features_list, 
                                #model_artefacts_dict, #lock
                                ):
    #if not use_objects_in_image_as_features:
    feat_extract = FeatureExtractor(seed=seed, img_resize_width=img_resize_width,
                                    img_resize_height=img_resize_height, 
                                    model_family=model_family,
                                    model_name=model_name,
                                    img_normalization_weight=img_normalization_weight
                                    )
    feat_extract.set_seed_consistently()
    model, preprocess = feat_extract.load_model_and_preprocess_func()
    feature_extractor = feat_extract.get_feature_extractor(model)
    img = feat_extract.load_and_resize_image(img_path, img_resize_width, img_resize_height)
    img_for_infer = feat_extract.load_image_for_inference(img_path, feat_extract.image_shape)
    feature = feat_extract.extract_features(img_for_infer, feature_extractor, preprocess)
    if return_img_path:
        return img, feature, img_path
    else:
        return img, feature

def get_imgs_and_extract_features_wrapper(args):
    img, feature, img_path = get_imgs_and_extract_features(**args)
    return img, feature, img_path
     
def extract_object_features_per_image(coco_annotation_filepath, img_dir,
                                      img_property_set: ImgPropertySetReturnType,
                                      coco=None, img_names=None
                                      ):#->Tuple[List, List]:
    if not coco:
        coco = COCO(coco_annotation_filepath)
    if not img_names:
        img_names = [obj["file_name"] for obj in coco.imgs.values()]
    if not isinstance(img_names, list):
        img_names = [img_names]
    #obj_featlist = []
    #imgname_list = []
    img_feature = {}
    for img in img_names:
        imgname = os.path.basename(img)
        objects = get_objects(imgname=imgname, coco=coco, img_dir=img_dir)
        features = get_object_features(obj_imgs=objects, seed=2024, img_resize_width=224,
                                        img_resize_height=224,
                                        model_family="efficientnet",
                                        model_name="EfficientNetB0",
                                        img_normalization_weight="imagenet",
                                        )
        #obj_featlist.append(features)
        #imgname_list.append(imgname)
        img_feature[imgname] = features
    img_property_set.img_names = [img_name for img_name in img_feature.keys()]
    img_property_set.features = [feat for feat in img_feature.values()]
    return img_property_set

def extract_object_features_per_image_wrapper(args):
    img_property_set = extract_object_features_per_image(**args)
    return img_property_set


def get_imgs_and_extract_features_multiprocess(img_path, img_resize_width,
                                               img_resize_height,
                                               model_family, model_name,
                                               img_normalization_weight,
                                               seed, images_list, features_list
                                               ):
    feat_extract = FeatureExtractor(seed=seed, img_resize_width=img_resize_width,
                                    img_resize_height=img_resize_height, 
                                    model_family=model_family,
                                    model_name=model_name,
                                    img_normalization_weight=img_normalization_weight
                                    )
    #images_list = []
    #features_list = []
    feat_extract.set_seed_consistently()
    model, preprocess = feat_extract.load_model_and_preprocess_func()
    feature_extractor = feat_extract.get_feature_extractor(model)
    img = feat_extract.load_and_resize_image(img_path, img_resize_width, img_resize_height)
    img_for_infer = feat_extract.load_image_for_inference(img_path, feat_extract.image_shape)
    feature = feat_extract.extract_features(img_for_infer, feature_extractor, preprocess)
    images_list.append(img)
    features_list.append(feature)
    print(f"total imgs processed {len(images_list)}")
    print(f"total features processed {len(features_list)}")
    #lock.release()
    return images_list, features_list

#%%
def img_feature_extraction_implementor(img_property_set,
                                       #feature_extractor_class = None,
                                       seed=2024, img_resize_width=224,
                                       img_resize_height=224,
                                       model_family="efficientnet",
                                       model_name="EfficientNetB0",
                                       img_normalization_weight="imagenet",
                                       #multiprocess = False
                                       ):
    
    img_paths = sorted(img_property_set.img_paths)
    img_list, feature_list = [], []
    for img_path in img_paths:
        img, feature = get_imgs_and_extract_features(img_path=img_path, 
                                                        img_resize_height=img_resize_height,
                                                    img_resize_width=img_resize_width,
                                                    model_family=model_family, 
                                                    model_name=model_name,
                                                    img_normalization_weight=img_normalization_weight,
                                                    seed=seed
                                                    )
        img_list.append(img)
        feature_list.append(feature)
    img_property_set.imgs = img_list
    img_property_set.features = feature_list
    return img_property_set
        
def run_multiprocess(img_property_set,
                    feature_extractor_class = None,
                    seed=2024, img_resize_width=224,
                    img_resize_height=224,
                    model_family="efficientnet",
                    model_name="EfficientNetB0",
                    img_normalization_weight="imagenet",
                    ):
    img_paths = sorted(img_property_set.img_paths)
    args = [{"img_path": img_path, "img_resize_width": img_resize_width,
                 "img_resize_height": img_resize_height, 
                 "model_family": model_family,
                 "model_name":model_name, 
                 "img_normalization_weight": img_normalization_weight,
                 "seed": seed, "return_img_path": True
                 } for img_path in img_paths
                ]
    chunksize = max(1, len(args) // 10)
    num_processes = multiprocessing.cpu_count()
    from tqdm import tqdm
    with multiprocessing.Pool(num_processes) as p:
        results = list(
                    tqdm(
                        p.imap_unordered(
                            get_imgs_and_extract_features_wrapper, args, chunksize=chunksize
                        ),
                        total=len(img_paths),
                    )
                )
    print("multiprocess of imaged feature extration completed")
    images_read = []
    features = []
    img_names = []
    img_paths = []
    print(f"started clustering")
    for res in results:
        images_read.append(res[0])
        features.append(res[1])
        img_names.append(os.path.basename(res[2]))
        img_paths.append(res[2])
    img_property_set.img_names = img_names
    img_property_set.img_paths = img_paths
    img_property_set.features = features
    return img_property_set
    # featarray = np.array(features)
    # ce = clusteval()
    # cluster_results = ce.fit(featarray)
    # clusters = cluster_results["labx"]
    # imgcluster_dict = {"image_names":image_names, "clusters": clusters}
    # imgclust_df = pd.DataFrame.from_dict(imgcluster_dict)
    # print("completed clustering")
    # return imgclust_df

    
#%%
if __name__ == '__main__':
    img_dir = "field_crop_with_disease"
    img_dir = "/home/lin/codebase/__cv_with_roboflow_data/field_crop_with_disease"
    img_paths_list = sorted(glob(f"{img_dir}/*"))
    img_names = [os.path.basename(img) for img in img_paths_list]
    img_property_set = ImgPropertySetReturnType(img_paths=img_paths_list, img_names=img_names, total_num_imgs=100, max_num_clusters=4)


    # img_property_set = img_feature_extraction_implementor(img_property_set=img_property_set,
    #                                                     use_cropped_imgs=False
    #                                                     )
    
    imgclust_df = run_multiprocess(img_property_set=img_property_set)
    
    #%%
    
    len(results)
    results[0]
    
     
    #%%
    featarray = np.array(img_property_set.features)
    ce = clusteval()
    results = ce.fit(featarray)
    clusters = results["labx"]
    imgcluster_dict = {"image_names": img_property_set.img_names, "clusters": clusters}
    imgclust_df = pd.DataFrame.from_dict(imgcluster_dict)
    
    
    #%%
    # Load COCO annotations
    img_dirs = "tomato_fruit"
    coco = COCO('coco_annotation_coco.json')


    img_paths = glob(f"{img_dirs}/*")
    obj_featlist = []
    for img in img_paths:
        imgname = os.path.basename(img)
        objects = get_objects(imgname=imgname, coco=coco, img_dir=img_dirs)
        features = get_object_features(obj_imgs=objects, seed=2024, img_resize_width=224,
                            img_resize_height=224,
                            model_family="efficientnet",
                            model_name="EfficientNetB0",
                            img_normalization_weight="imagenet",
                            )
        obj_featlist.append(features)
        
    #%%

    obj_featlist[0] == obj_featlist[1]

    #%%
    obj_featlist[1]
    #%%


    image = cv2.imread(img_paths[0])

    #%%
    Image.fromarray(image)

    img_tensor = tf.convert_to_tensor(image)

    #%%
    # TensorShape([1032, 774, 3])
    img_resize_with_pad = tf.image.resize_with_pad(img_tensor, target_height=100, target_width=100)
    #Image.fromarray(img_tensor)

    tf.image.resize_with_pad()
    img_array = img_resize_with_pad.numpy().astype(np.uint8)
    Image.fromarray(img_array)

    #%%
    np.array(img_resize_with_pad).shape
    #%%
    image = tf.io.read_file(img_paths[0])   

    #%%
    read_img_list, feat_list = [], []
    for img in img_paths_list:
        read_img, feat_ext = get_imgs_and_extract_features(img_path=img, 
                                                        seed=2024, img_resize_width=224,
                                                        img_resize_height=224,
                                                        model_family="efficientnet",
                                                        model_name="EfficientNetB0",
                                                        img_normalization_weight="imagenet",
                                                        )
        read_img_list.append(read_img)
        feat_list.append(feat_ext)


