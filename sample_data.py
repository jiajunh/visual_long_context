import os
import json
import random
import shutil


def load_json_file(data_path):
    with open(data_path) as f:
        data = json.load(f)
    return data


if __name__ == "__main__":
    cap_data_path = "./data/coco/annotations/captions_train2017.json"
    instance_data_path = "./data/coco/annotations/instances_train2017.json"
    img_path = "./data/coco/train2017"
    output_dir = "./data/samples_500"
    num_samples = 500

    categories = {}
    full_data = {}
    samples = {}

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)

    img_files = os.listdir(img_path)
    sampled_items = sorted(random.sample(img_files, num_samples))

    cap_data = load_json_file(cap_data_path)
    # print(len(cap_data), type(cap_data))
    print(cap_data.keys())

    instance_data = load_json_file(instance_data_path)
    # print(len(instance_data), type(instance_data))
    print(instance_data.keys())

    
    category_data = instance_data["categories"]
    for cat in category_data:
        categories[cat["id"]] = cat["name"]
    
    images = instance_data["images"]

    for img in images:
        full_data[img["id"]] = {
            "url": img["coco_url"],
            "file_name": img["file_name"],
            "captions": [],
            "categories": set(),
        }
    
    for ann in instance_data["annotations"]:
        full_data[ann["image_id"]]["categories"].add(categories[ann["category_id"]])
    
    for ann in cap_data["annotations"]:
        full_data[ann["image_id"]]["captions"].append(ann["caption"])


    for img_file in sampled_items:
        src_path = os.path.join(img_path, img_file)
        dst_path = os.path.join(output_dir, "images", img_file)
        shutil.copy2(src_path, dst_path)

        img_idx = int(img_file.split("_")[-1][:-4])

        samples[img_idx] = {
            "file_name": full_data[img_idx]["file_name"],
            "url": full_data[img_idx]["url"],
            "categories": list(full_data[img_idx]["categories"]),
            "captions": full_data[img_idx]["captions"],
        }

    with open(os.path.join(output_dir, "captions.json"), "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False)

