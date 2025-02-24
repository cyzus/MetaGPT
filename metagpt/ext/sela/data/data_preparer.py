from metagpt.ext.sela.utils import DATA_CONFIG, DATASET_CONFIG
from metagpt.ext.sela.data.dataset import ExpDataset, save_datasets_dict_to_yaml
from pathlib import Path
import os
import shutil


class DatasetPreparer:
    def __init__(self, dataset_name, target_col, data_path, datasets_dir=DATA_CONFIG["datasets_dir"], dataset_dict=None):
        """Prepare tabular data for SELA"""
        self.dataset_name = dataset_name
        self.datasets_dir = datasets_dir
        self.target_col = target_col
        self.data_path = data_path
        self.datasets_dict = DATASET_CONFIG
        self.dataset_dict = dataset_dict

    def create_data_structure(self):
        raw_dir = Path(self.datasets_dir, self.dataset_name, "raw")
        data_path = Path(self.data_path)
        assert data_path.name.endswith(".csv"), "The dataset must be a csv file"
        new_path = Path(raw_dir, "train.csv")
        if not os.path.exists(raw_dir):
            os.makedirs(raw_dir)
            # copy dataset (a file) to raw_dir
            shutil.copy(data_path, new_path)

    def get_dataset_dict(self):
        return self.dataset_dict

    def prepare_dataset(self):
        self.create_data_structure()
        custom_dataset = ExpDataset(self.dataset_name, self.datasets_dir, target_col=self.target_col, force_update=True)
        if self.dataset_dict is not None:
            user_requirement = self.dataset_dict["user_requirement"]
            self.dataset_dict.update(custom_dataset.create_dataset_dict())
            self.dataset_dict["user_requirement"] = user_requirement
        self.datasets_dict["datasets"][self.dataset_name] = self.dataset_dict
        save_datasets_dict_to_yaml(self.datasets_dict)


if __name__ == "__main__":
    dataset_path = "C:/Users/STGF/Documents/workspace/datasets/dpas_st/confidence/train.csv"
    dataset_preparer = DatasetPreparer("confidence_06", "E2E_label", dataset_path)
    dataset_preparer.prepare_dataset()



