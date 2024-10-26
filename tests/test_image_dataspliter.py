from image_dataspliter.clust import object_based_cluster_images_from_cocoann
from image_dataspliter.generate_coco_ann import generate_random_images_and_annotation
import pytest
import tempfile

@pytest.fixture
def generate_crop_imgs_and_annotation():
    tempdir = tempfile.TemporaryDirectory()
    img_paths, gen_coco_path = generate_random_images_and_annotation(image_height=124, image_width=124,
                                                                    number_of_images=10, 
                                                                    output_dir=tempdir.name,
                                                                    img_ext=None,
                                                                    image_name=None,
                                                                    parallelize=True
                                                                    )
    return img_paths, gen_coco_path, tempdir


def test_object_based_cluster_images_from_cocoann():
    pass