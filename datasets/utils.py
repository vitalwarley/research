import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import default_collate


def sr_collate_fn(batch):
    """
    Collate function for Search and Retrieval.
    """

    # Unpack the batch
    (probe_index, probe_images), (gallery_indexes, gallery_images) = batch[0]

    # Convert probe_index to a tensor
    probe_index = torch.tensor([probe_index])

    # Concatenate probe_images and gallery_images
    # Transform list of tensors into a single tensor per each
    probe_images_tensor = torch.stack(
        probe_images
    )  # Shape: [num_probe_images, 3, 112, 112]
    gallery_images_tensor = torch.stack(
        gallery_images
    )  # Shape: [num_gallery_images, 3, 112, 112]

    # Handle gallery_indexes which is a list of tensors
    # Since gallery_indexes are used for indexing or referencing, they could be concatenated as well
    gallery_indexes_tensor = torch.tensor(gallery_indexes)

    return (probe_index, probe_images_tensor), (
        gallery_indexes_tensor,
        gallery_images_tensor,
    )


def sr_collate_fn_v2(batch):
    """
    Collate function for Search and Retrieval V2.

    Keep only first probe tensor.
    """

    # Unpack the batch
    probe_info, gallery_info = batch[0]  # Always 1 element
    probe_id, probe_images = probe_info
    gallery_ids, gallery_images = zip(
        *[(gallery_id, gallery_images) for (gallery_id, gallery_images) in gallery_info]
    )

    return (
        probe_id,
        default_collate(probe_images),
        default_collate(gallery_ids),
        default_collate(gallery_images),
    )


def collate_fn_fiw_family_v3(batch):
    # Unpack the batch - each item is (images, labels) where:
    # images = (imgs1, imgs2)
    # labels = (is_kin, kin_ids)
    imgs1_batch = [item[0][0] for item in batch]
    imgs2_batch = [item[0][1] for item in batch]
    kin_ids = [item[1][0] for item in batch]
    is_kin = [item[1][1] for item in batch]

    # Flatten the list of lists into a single list of tensors
    imgs1_flat = [img for imgs in imgs1_batch for img in imgs]
    imgs2_flat = [img for imgs in imgs2_batch for img in imgs]
    is_kin_flat = [label for labels in is_kin for label in labels]
    kin_ids_flat = [kid for kids in kin_ids for kid in kids]

    # Stack tensors along the batch dimension
    imgs1_tensor = torch.stack(imgs1_flat)
    imgs2_tensor = torch.stack(imgs2_flat)
    is_kin_tensor = torch.tensor(is_kin_flat)
    kin_ids_tensor = [Sample.NAME2LABEL[kid] for kid in kin_ids_flat]
    kin_ids_tensor = torch.tensor(kin_ids_tensor)

    return (imgs1_tensor, imgs2_tensor), (kin_ids_tensor, is_kin_tensor)


def collate_fn_fiw_family_v3_task2(batch):
    """Collate function for Task 2 (tri-subject verification) that handles triplets.

    Similar to collate_fn_fiw_family_v3 but handles father, mother, child triplets.

    Args:
        batch: List of tuples (images, labels) where:
            images = (father_imgs, mother_imgs, child_imgs)
            labels = (kin_ids, is_kin)

    Returns:
        tuple: ((father_tensor, mother_tensor, child_tensor), (kin_ids_tensor, is_kin_tensor))
    """
    # Unpack the batch
    father_batch = [item[0][0] for item in batch]  # List of father images
    mother_batch = [item[0][1] for item in batch]  # List of mother images
    child_batch = [item[0][2] for item in batch]  # List of child images
    kin_ids = [item[1][0] for item in batch]  # List of kinship types
    is_kin = [item[1][1] for item in batch]  # List of kinship labels

    # Flatten the list of lists into a single list of tensors
    father_flat = [img for imgs in father_batch for img in imgs]
    mother_flat = [img for imgs in mother_batch for img in imgs]
    child_flat = [img for imgs in child_batch for img in imgs]
    is_kin_flat = [label for labels in is_kin for label in labels]
    kin_ids_flat = [kid for kids in kin_ids for kid in kids]

    # Stack tensors along the batch dimension
    father_tensor = torch.stack(father_flat)
    mother_tensor = torch.stack(mother_flat)
    child_tensor = torch.stack(child_flat)
    is_kin_tensor = torch.tensor(is_kin_flat)
    kin_ids_tensor = [
        SampleTask2.NAME2LABEL[kid] for kid in kin_ids_flat
    ]  # Use Task2 labels
    kin_ids_tensor = torch.tensor(kin_ids_tensor)

    return (father_tensor, mother_tensor, child_tensor), (kin_ids_tensor, is_kin_tensor)


def collate_fn_fiw_family_v4(batch):
    imgs1_batch = [item[0] for item in batch]
    imgs2_batch = [item[1] for item in batch]
    is_kin = [item[2][-1] for item in batch]
    imgs1_age = [item[2][0] for item in batch]
    imgs2_age = [item[2][1] for item in batch]
    imgs1_gender = [item[2][2] for item in batch]
    imgs2_gender = [item[2][3] for item in batch]

    # Flatten the list of lists into a single list of tensors
    imgs1_flat = [img for imgs in imgs1_batch for img in imgs]
    imgs2_flat = [img for imgs in imgs2_batch for img in imgs]
    is_kin_flat = [label for labels in is_kin for label in labels]
    imgs1_age_flat = [label for labels in imgs1_age for label in labels]
    imgs2_age_flat = [label for labels in imgs2_age for label in labels]
    imgs1_gender_flat = [label for labels in imgs1_gender for label in labels]
    imgs2_gender_flat = [label for labels in imgs2_gender for label in labels]

    # Stack tensors along the batch dimension
    imgs1_tensor = torch.stack(imgs1_flat)
    imgs2_tensor = torch.stack(imgs2_flat)
    is_kin_tensor = torch.tensor(is_kin_flat)
    imgs1_age_tensor = torch.tensor(imgs1_age_flat)
    imgs2_age_tensor = torch.tensor(imgs2_age_flat)
    imgs1_gender_tensor = torch.tensor(imgs1_gender_flat)
    imgs2_gender_tensor = torch.tensor(imgs2_gender_flat)
    labels = (
        imgs1_age_tensor,
        imgs2_age_tensor,
        imgs1_gender_tensor,
        imgs2_gender_tensor,
        is_kin_tensor,
    )

    return imgs1_tensor, imgs2_tensor, labels


# Example usage:
# dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)


class Sample:
    # TODO: move to utils.py
    NAME2LABEL = {
        "non-kin": 0,
        "md": 1,
        "ms": 2,
        "sibs": 3,
        "ss": 4,
        "bb": 5,
        "fd": 6,
        "fs": 7,
        "gfgd": 8,
        "gfgs": 9,
        "gmgd": 10,
        "gmgs": 11,
    }

    def __init__(
        self, id: str, f1: str, f2: str, kin_relation: str, is_kin: str, *args, **kwargs
    ):
        self.id = id
        self.f1 = f1
        self.f2 = f2
        self.kin_relation = kin_relation
        self.is_kin = int(is_kin)
        self.is_same_generation = self.kin_relation in ["bb", "ss", "sibs"]
        f1_parts = f1.split("/")
        f2_parts = f2.split("/")
        self.set_fids(f1_parts, f2_parts)
        self.set_mids(f1_parts, f2_parts)

    def set_fids(self, f1, f2):
        try:
            self.f1fid = int(f1[2][1:])
            self.f2fid = int(f2[2][1:])
        except Exception:
            self.f1fid = 0
            self.f2fid = 0

    def set_mids(self, f1, f2):
        try:
            self.f1mid = int(f1[3][3:])
            self.f2mid = int(f2[3][3:])
        except Exception:
            self.f1mid = 0
            self.f2mid = 0


class SampleTask2:
    # TODO: move to utils.py
    NAME2LABEL = {
        "non-kin": 0,
        "FMD": 1,
        "FMS": 2,
    }

    def __init__(
        self, f1: str, f2: str, f3: str, kin_relation: str, is_kin: str, *args, **kwargs
    ):
        self.f1 = f1.lstrip("./")  # father
        self.f2 = f2.lstrip("./")  # mother
        self.f3 = f3.lstrip("./")  # child
        self.kin_relation = kin_relation
        self.is_kin = int(is_kin)
        f1_parts = f1.lstrip("./").split("/")
        f2_parts = f2.lstrip("./").split("/")
        f3_parts = f3.lstrip("./").split("/")
        self.set_fids(f1_parts, f2_parts, f3_parts)
        self.set_mids(f1_parts, f2_parts, f3_parts)

    def set_fids(self, f1, f2, f3):
        try:
            self.f1fid = int(f1[2][1:])  # father's family ID
            self.f2fid = int(f2[2][1:])  # mother's family ID
            self.f3fid = int(f3[2][1:])  # child's family ID
        except Exception:
            self.f1fid = 0
            self.f2fid = 0
            self.f3fid = 0

    def set_mids(self, f1, f2, f3):
        try:
            self.f1mid = int(f1[3][3:])  # father's member ID
            self.f2mid = int(f2[3][3:])  # mother's member ID
            self.f3mid = int(f3[3][3:])  # child's member ID
        except Exception:
            self.f1mid = 0
            self.f2mid = 0
            self.f3mid = 0


class SampleProbe:
    def __init__(self, id: str, s1_dir: str):
        self.id = int(id)
        self.s1_dir = s1_dir


class SampleGallery:
    def __init__(self, id: str, f1: str):
        self.id = int(id)
        self.f1 = f1


def one_hot_encode_kinship(relation):
    index = Sample.NAME2LABEL[relation] - 1
    one_hot = torch.zeros(len(Sample.NAME2LABEL))
    one_hot[index] = 1
    return one_hot


def read_image(path: Path, shape=(112, 112)):
    # TODO: add to utils.py
    img = cv2.imread(str(path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, shape)
    return img


def worker_init_fn(worker_id):
    # Calculate unique seed for each worker
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
