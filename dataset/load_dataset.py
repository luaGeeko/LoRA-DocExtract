import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List
from src.utils.logger import setup_logger


class SROIEDataset:
    def __init__(self, dataset_dir: Path, debug: Optional[bool] = False, manifest: Optional[List[str]] = None):
        self.dataset_dir = Path(dataset_dir)
        self.manifest = manifest or ['train', 'eval', 'test']
        self.logger = setup_logger(self.__class__.__name__, debug=debug)

        # populate the dataset with proper splits
        # self.train = None
        # self.val = None
        # self.test = None
        self.load_dataset()

    def _validate_files_count(self, path: Path, split: str):
        img_dir = path / "img"
        box_dir = path / "box"
        ent_dir = path / "entities"
        if not img_dir.exists() or not box_dir.exists() or not ent_dir.exists():
            self.logger.warning(f"Missing subfolders for split {split}: {path}")
            return set()
        # file stems without extension
        img_files = {p.stem for p in img_dir.iterdir() if p.is_file()}
        box_files = {p.stem for p in box_dir.iterdir() if p.is_file()}
        ent_files = {p.stem for p in ent_dir.iterdir() if p.is_file()}
        self.logger.debug(f"Images   : {len(img_files)}")
        self.logger.debug(f"Boxes    : {len(box_files)}")
        self.logger.debug(f"Entities : {len(ent_files)}")
        # any missing box or entities files
        missing_boxes = img_files - box_files
        missing_entities = img_files - ent_files
        if missing_boxes:
            self.logger.warning(f"Missing box files: {len(missing_boxes)} corresponding paths will not included in dataset")
        if missing_entities:
            self.logger.warning(f"Missing entity files: {len(missing_entities)} corresponding paths will not included in dataset")
        img_files = img_files - missing_boxes - missing_entities
        self.logger.info(f"Images with all files taken for {split}: {len(img_files)}")
        return img_files

    def _load_split(self, split: str):
        if split not in self.manifest:
            self.logger.warning(f"Split {split} not in manifest, skipping loading this split....")
            return
        full_path = self.dataset_dir / split
        valid_img_files = self._validate_files_count(path=full_path, split=split)
        if len(valid_img_files) == 0:
            self.logger.warning(f"No valid samples found for split {split}")
            return None
        rows = []
        for img in valid_img_files:
            box_path = full_path / "box" / f"{img}.txt"
            ent_path = full_path / "entities" / f"{img}.txt"
            rows.append({
                "doc_id": img,
                "img_path": str(full_path / "img" / f"{img}.jpg"),
                "box_path": str(box_path),
                "ent_path": str(ent_path),
            })
        data = pd.DataFrame(rows)
        return data

    def load_dataset(self):
        # load the dataset from each split in the manifest and then format in pandas format
        for split in self.manifest:
            setattr(self, split, self._load_split(split=split))


# DATASET_DIR = Path("/Users/lua/.cache/kagglehub/datasets/dattrinh12/sroie-dataset")
# data_splits_path = DATASET_DIR / "versions/1"
# sroie_dataset = SROIEDataset(dataset_dir=data_splits_path, debug=True)