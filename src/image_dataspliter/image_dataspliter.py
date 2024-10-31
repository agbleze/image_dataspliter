
from image_dataspliter.feat import ImgPropertySetReturnType
import inspect
from sklearn.model_selection import train_test_split
import numpy as np
from image_dataspliter.clust import (object_based_cluster_images_from_cocoann,
                                    cluster_objects_with_added_features,
                                    cluster_with_full_image,
                                    clusters_with_full_image_multiprocess,
                                    object_based_cluster_images_from_cocoann_multiprocess,
                                    cluster_objects_with_added_features_multiprocess
                                    )

def get_cluster_func(use_object_features, parallelize, insitu, **kwargs):
    if not use_object_features and not parallelize:
        func = cluster_with_full_image
    elif not use_object_features and parallelize:
        func = clusters_with_full_image_multiprocess
    elif use_object_features and insitu and not parallelize:
        func = object_based_cluster_images_from_cocoann
    elif use_object_features and insitu and parallelize:
        func = object_based_cluster_images_from_cocoann_multiprocess
    elif use_object_features and not insitu and not parallelize:
        func = cluster_objects_with_added_features
    elif use_object_features and not insitu and parallelize:
        func = cluster_objects_with_added_features_multiprocess
    return func
        
def get_params(func, kwargs):
    allowed_param = [param for param in kwargs 
                    if param in 
                    inspect.signature(func).parameters
                    ]
    useparams = {param: kwargs[param] for param in 
                 allowed_param
                 }
    return useparams
           
          
def split_data(*args, **kwargs):
    func_params = get_params(func=get_cluster_func, kwargs=kwargs)
    cluster_func = get_cluster_func(**func_params)
    params_to_use = get_params(cluster_func, kwargs=kwargs)
    cluster_df = cluster_func(**params_to_use)
    include_testsplit = kwargs.get('include_testsplit', True)
    train_size = kwargs.get('train_size', 0.9)
    train_df, test_df = train_test_split(cluster_df, train_size=train_size,
                                        stratify=cluster_df.clusters,
                                        random_state=2024
                                        )
    if not include_testsplit:
        results = {"train_set": train_df.image_names.values.tolist(),
                    "val_set": test_df.image_names.values.tolist()
                    }
        cluster_df["split_type"] = np.where(cluster_df.image_names.isin(train_df.image_names.values.tolist()), 
                                            "train", "nosplit"
                                            )
        cluster_df["split_type"] = np.where(cluster_df.image_names.isin(test_df.image_names.values.tolist()), 
                                            "val", cluster_df.split_type
                                            )
        cluster_df.to_csv("data_split.csv")
    elif include_testsplit:
        train_df, val_df = train_test_split(train_df, train_size=train_size,
                                            stratify=train_df.clusters,
                                            random_state=2024
                                            )
        cluster_df["split_type"] = np.where(cluster_df.image_names.isin(train_df.image_names.values.tolist()), 
                                            "train", "nosplit"
                                            )
        cluster_df["split_type"] = np.where(cluster_df.image_names.isin(test_df.image_names.values.tolist()), 
                                            "test", cluster_df.split_type
                                            )
        cluster_df["split_type"] = np.where(cluster_df.image_names.isin(val_df.image_names.values.tolist()), 
                                            "val", cluster_df.split_type
                                            )
        cluster_df.to_csv("data_split.csv")
        
        results = {"train_set": train_df.image_names.values.tolist(),
                    "val_set": val_df.image_names.values.tolist(),
                    "test_set": test_df.image_names.values.tolist()
                    }
        
    return results
    