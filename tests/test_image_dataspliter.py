import pytest
import tempfile
import pandas as pd
import os
from image_dataspliter.generate_coco_ann import generate_random_images_and_annotation
from image_dataspliter.feat import ImgPropertySetReturnType
from image_dataspliter.clust import (object_based_cluster_images_from_cocoann,
                                    cluster_objects_with_added_features,
                                    cluster_with_full_image,
                                    clusters_with_full_image_multiprocess,
                                    object_based_cluster_images_from_cocoann_multiprocess,
                                    cluster_objects_with_added_features_multiprocess
                                    )
from image_dataspliter.image_dataspliter import split_data

@pytest.fixture
def generate_crop_imgs_and_annotation():
    tempdir = tempfile.TemporaryDirectory()
    img_paths, gen_coco_path = generate_random_images_and_annotation(image_height=124, image_width=124,
                                                                    number_of_images=200, 
                                                                    output_dir=tempdir.name,
                                                                    img_ext=None,
                                                                    image_name=None,
                                                                    parallelize=True
                                                                    )
    return img_paths, gen_coco_path, tempdir

@pytest.fixture
def img_property_set():
    imgset = ImgPropertySetReturnType(img_names=None, img_paths=None, 
                                      total_num_imgs=None, 
                                      max_num_clusters=None
                                      )
    return imgset

def check_cluster_df(cluster_df):
    assert isinstance(cluster_df, pd.DataFrame), f"cluster results is {type(cluster_df)} but pd.DataFrame is expected"
    for col in ["image_names", "clusters"]:
        assert col in cluster_df.columns, f"Column name {col} not found in cluster results"

def test_split_data(generate_crop_imgs_and_annotation, img_property_set,
                    use_object_features=False, 
                    parallelize=False, 
                    insitu=False,
                    include_testsplit=True
                    ):
    img_paths, coco_path, tempdir = generate_crop_imgs_and_annotation
    img_property_set = img_property_set
    img_names = [os.path.basename(img_path) for img_path in img_paths]
    total_num_imgs = len(img_paths)
    img_property_set.img_names = img_names
    img_property_set.img_paths = img_paths
    img_property_set.total_num_imgs = total_num_imgs
    results = split_data(img_property_set=img_property_set,
                         use_object_features=use_object_features, 
                        parallelize=parallelize, 
                        insitu=insitu, 
                        include_testsplit=include_testsplit
                        )
    assert isinstance(results, dict), f"split_data result is not a dict"
    for datasplit in ["train_set", "val_set", "test_set"]:
        assert datasplit in results, f"{datasplit} not found in split_data results"
        assert results[datasplit] is not None, f"{datasplit} is empty in split_data results"


def test_object_based_cluster_images_from_cocoann(generate_crop_imgs_and_annotation, img_property_set):
    img_paths, coco_path, tempdir = generate_crop_imgs_and_annotation
    img_property_set = img_property_set
    cluster_df = object_based_cluster_images_from_cocoann(coco_annotation_file=coco_path,
                                             img_dir=tempdir.name,
                                             img_property_set=img_property_set
                                             )
    check_cluster_df(cluster_df)
    tempdir.cleanup()
    if os.path.exists(coco_path):
        os.remove(coco_path)
        
def test_cluster_objects_with_added_features(generate_crop_imgs_and_annotation, img_property_set):
    img_paths, coco_path, tempdir = generate_crop_imgs_and_annotation
    img_property_set = img_property_set
    cluster_df = cluster_objects_with_added_features(img_dir=tempdir.name, 
                                                    coco_annotation_filepath=coco_path,
                                                    img_property_set=img_property_set
                                                    )
    check_cluster_df(cluster_df)
    tempdir.cleanup()
    if os.path.exists(coco_path):
        os.remove(coco_path)


def test_cluster_with_full_image(generate_crop_imgs_and_annotation, img_property_set):
    img_paths, coco_path, tempdir = generate_crop_imgs_and_annotation
    img_property_set = img_property_set
    img_names = [os.path.basename(img_path) for img_path in img_paths]
    total_num_imgs = len(img_paths)
    img_property_set.img_names = img_names
    img_property_set.img_paths = img_paths
    img_property_set.total_num_imgs = total_num_imgs
    cluster_df = cluster_with_full_image(img_property_set=img_property_set)
    check_cluster_df(cluster_df)
    tempdir.cleanup()
    if os.path.exists(coco_path):
        os.remove(coco_path)
        
        
def test_clusters_with_full_image_multiprocess(generate_crop_imgs_and_annotation, img_property_set):
    img_paths, coco_path, tempdir = generate_crop_imgs_and_annotation
    img_property_set = img_property_set
    img_names = [os.path.basename(img_path) for img_path in img_paths]
    total_num_imgs = len(img_paths)
    img_property_set.img_names = img_names
    img_property_set.img_paths = img_paths
    img_property_set.total_num_imgs = total_num_imgs
    cluster_df = clusters_with_full_image_multiprocess(img_property_set=img_property_set)
    check_cluster_df(cluster_df)
    tempdir.cleanup()
    if os.path.exists(coco_path):
        os.remove(coco_path)
        
def test_object_based_cluster_images_from_cocoann_multiprocess(generate_crop_imgs_and_annotation, 
                                                               img_property_set
                                                               ):
    img_paths, coco_path, tempdir = generate_crop_imgs_and_annotation
    img_property_set = img_property_set
    cluster_df = object_based_cluster_images_from_cocoann_multiprocess(coco_annotation_file=coco_path,
                                                                        img_dir=tempdir.name,
                                                                        img_property_set=img_property_set
                                                                        )
    check_cluster_df(cluster_df)
    tempdir.cleanup()
    if os.path.exists(coco_path):
        os.remove(coco_path)
        
def test_cluster_objects_with_added_features_multiprocess(generate_crop_imgs_and_annotation, img_property_set):
    img_paths, coco_path, tempdir = generate_crop_imgs_and_annotation
    img_property_set = img_property_set
    cluster_df = cluster_objects_with_added_features_multiprocess(coco_annotation_filepath=coco_path,
                                                                img_dir=tempdir.name,
                                                                )
    check_cluster_df(cluster_df)
    tempdir.cleanup()
    if os.path.exists(coco_path):
        os.remove(coco_path)
   




      