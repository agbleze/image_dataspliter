
#%%
import json
import random
import os
from randimage import get_random_image
from tqdm import tqdm
import matplotlib
from glob import glob
import multiprocessing

def generate_random_bbox(image_width, image_height):
    x = random.randint(0, image_width - 1)
    y = random.randint(0, image_height - 1)
    width = random.randint(1, image_width - x)
    height = random.randint(1, image_height - y)
    return [x, y, width, height]

def generate_random_segmentation(bbox):
    x, y, width, height = bbox
    points = [
        x, y,
        x + width, y,
        x + width, y + height,
        x, y + height
    ]
    return [points]

def create_coco_annotation(image_id, category_id, bbox, segmentation, ann_id):
    annotation = {
        "id": ann_id,
        "image_id": image_id,
        "category_id": category_id,
        "bbox": bbox,
        "segmentation": segmentation,
        "area": bbox[2] * bbox[3],
        "iscrowd": 0
    }
    return annotation

def generate_coco_annotation_file(image_width, image_height, output_path, img_list):
    images, annotations, categories = [], [], []
    if not img_list:
        raise ValueError(f"img_list is required to be a list of str or path but {img_list} was given")
    for idx, img_path in enumerate(img_list):
        category_id = random.sample(population=[1,2,3], k=1)[0]
        image_id = idx + 1
        bbox = generate_random_bbox(image_width, image_height)
        segmentation = generate_random_segmentation(bbox)
        annotation = create_coco_annotation(image_id, category_id, bbox, segmentation,
                                            ann_id=image_id
                                            )
        img_info = {"id": image_id,
                    "width": image_width,
                    "height": image_height,
                    "file_name": os.path.basename(img_path)
                }
        category_info = {"id": category_id,
                        "name": f"object_{category_id}",
                        "supercategory": "none"
                        }
        images.append(img_info)
        categories.append(category_info)
        annotations.append(annotation)
        
    coco_format = {"images": images,
                    "annotations": annotations,
                    "categories": categories
                    }
    
    with open(output_path, 'w') as f:
        json.dump(coco_format, f, indent=4)

def save_random_imgs(img_size, save_as):
    img = get_random_image(img_size)
    matplotlib.image.imsave(save_as, img)
    
def save_random_img_wrapper(args):
    save_random_imgs(**args)
    
def generate_random_images(image_height, image_width,
                            number_of_images, output_dir=None,
                            img_ext=None,
                            image_name=None,
                            parallelize=True
                            ):
    if not output_dir:
        output_dir = "random_images"
    if not image_name:
        image_name = "random_image"
    if not img_ext:
        img_ext = "jpg"
    os.makedirs(output_dir, exist_ok=True)
    img_size = (int(image_height), int(image_width))
    iterations = [i for i in range(0, int(number_of_images))]
    
    if not parallelize:
        for idx in tqdm(iterations, total=len(iterations), desc="Generating images"):
            save_as = os.path.join(output_dir, f"{image_name}_{str(idx)}.{img_ext}")
            save_random_imgs(img_size, save_as)
    else:
        args = [{"img_size": img_size,
                 "save_as": os.path.join(output_dir, f"{image_name}_{str(idx)}.{img_ext}")
                 } for idx in iterations
                ]
        chunksize_divider = 50
        chunksize = max(1, len(args)//chunksize_divider)
        num_cpus = multiprocessing.cpu_count()
        with multiprocessing.Pool(num_cpus) as p:
            list(tqdm(p.imap_unordered(save_random_img_wrapper,
                                       args,
                                       chunksize=chunksize
                                       ),
                      total=len(iterations),
                      desc="Generating images in multiprocessing"
                      )
                 )
    img_paths =  glob(f"{output_dir}/*")
    return img_paths       

def generate_random_images_and_annotation(image_height, image_width,
                                        number_of_images, output_dir=None,
                                        img_ext=None,
                                        image_name=None,
                                        parallelize=True,
                                        save_ann_as="generated_annotation.json"
                                        ):            
    img_paths = generate_random_images(image_height=image_height, 
                                       image_width=image_width,
                                        number_of_images=number_of_images, 
                                        output_dir=output_dir,
                                        img_ext=img_ext,
                                        image_name=image_name,
                                        parallelize=parallelize
                                        )
    generate_coco_annotation_file(image_width=image_width, 
                                  image_height=image_height, 
                                  output_path=save_ann_as, 
                                  img_list=img_paths
                                  )
    return img_paths, save_ann_as
        
    