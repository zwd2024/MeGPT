from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader
from dataset.data_helper import create_datasets

class DataModule(LightningDataModule):

    def __init__(
            self,
            args
    ):
        super().__init__()
        self.args = args

    def prepare_data(self):
        """
        使用此方法可以完成可能写入磁盘的操作，或者只需要在分布式设置中由单个进程完成的操作。

        下载

        tokenize

        etc…
        :return:
        """

    def setup(self, stage: str):
        """
        您可能还需要在每个GPU上执行数据操作。使用setup可以做这样的事情：

        统计类数

        建立的词库

        执行 train/val/test splits

        应用转换（在数据模块中显式定义或在init中分配）

        etc…
        :param stage:
        :return:
        """
        train_dataset, dev_dataset, test_dataset = create_datasets(self.args)
        self.dataset = {
            "train": train_dataset, "validation": dev_dataset, "test": test_dataset
        }


    def train_dataloader(self):
        """
        使用此方法生成列车数据加载器。通常你只是包装你在setup中定义的数据集。
        :return:
        """
        loader = DataLoader(self.dataset["train"], batch_size=self.args.batch_size, drop_last=True, pin_memory=True,
                        num_workers=self.args.num_workers, prefetch_factor=self.args.prefetch_factor)
        return loader


    def val_dataloader(self):
        """
        使用此方法生成val数据加载器。通常你只是包装你在setup中定义的数据集。
        :return:
        """
        loader = DataLoader(self.dataset["validation"], batch_size=self.args.val_batch_size, drop_last=False, pin_memory=True,
                            num_workers=self.args.num_workers, prefetch_factor=self.args.prefetch_factor)
        return loader


    def test_dataloader(self):
        loader = DataLoader(self.dataset["test"], batch_size=self.args.test_batch_size, drop_last=False, pin_memory=False,
                        num_workers=self.args.num_workers, prefetch_factor=self.args.prefetch_factor)
        return loader