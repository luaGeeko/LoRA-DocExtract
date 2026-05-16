import pandas as pd
import numpy as np
import kagglehub
from PIL import Image
from pathlib import Path
from typing import Optional, List, Union
from src.utils.logger import setup_logger


class SROIEDataset:
    def __init__(self, dataset_dir: Union[Path, str] = None, debug: Optional[bool] = False, manifest: Optional[List[str]] = None, resize_offline: Optional[bool] = False, max_edge: Optional[int] = 1024):
        self.manifest = manifest or ['train', 'eval', 'test']
        self.resize_offline = resize_offline
        self.max_edge = max_edge
        self.logger = setup_logger(self.__class__.__name__, debug=debug)
        if dataset_dir is None:
            self.logger.info("No dataset_dir provided. Sourcing from Kaggle...")
            self.dataset_dir = self._download_from_kaggle()
        else:
            self.dataset_dir = Path(dataset_dir)

        # setup dataset attributes for each split in manifest
        self.load_dataset()

    def _download_from_kaggle(self) -> Path:
        """
        Handles the downloading and caching of the dataset via kagglehub.
        Returns the path to the specific version folder.
        """
        try:
            # kagglehub.dataset_download automatically manages the cache and returns the direct path to the latest version (e.g., .../versions/1)
            download_path = kagglehub.dataset_download("dattrinh12/sroie-dataset")
            self.logger.info(f"Dataset successfully downloaded/located at: {download_path}")
            return Path(download_path)
        except Exception as e:
            self.logger.error(f"Failed to fetch dataset from Kaggle. Ensure kagglehub is authenticated: {e}")
            raise

    @staticmethod
    def resize_and_save_image(original_path: Path, resized_dir: Path, img_name: str, max_edge:int) -> Path:
        """ Handles the actual offline resizing and saving of a single image, returning the path to the resized image. """
        resized_dir.mkdir(parents=True, exist_ok=True)
        resized_path = resized_dir / f"{img_name}.jpg"
        # skip if already resized from a previous run
        if resized_path.exists():
            return resized_path
    
        with Image.open(original_path) as img:
            img = img.convert("RGB")
            img.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
            img.save(resized_path, format="JPEG", quality=85)
        return resized_path

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
        #NOTE: create resized image dir, these images directory is just for training, if you want to inspect orginal iamges with boxes, this will be taken from original img path, not resized path
        resized_dir = full_path / "img_resized"
        rows = []
        for img in valid_img_files:
            original_img_path = full_path / "img" / f"{img}.jpg"
            box_path = full_path / "box" / f"{img}.txt"
            ent_path = full_path / "entities" / f"{img}.txt"
            if self.resize_offline:
                final_img_path = self.resize_and_save_image(original_img_path, resized_dir, img, self.max_edge)
            else:
                final_img_path = original_img_path
            rows.append({
                "doc_id": img,
                "img_path": str(final_img_path),
                "box_path": str(box_path),
                "ent_path": str(ent_path),
            })
        data = pd.DataFrame(rows)
        # attach global metadata to the DataFrame
        data.attrs['is_resized_offline'] = self.resize_offline
        return data

    def load_dataset(self):
        # load the dataset from each split in the manifest and then format in pandas format
        for split in self.manifest:
            setattr(self, split, self._load_split(split=split))
