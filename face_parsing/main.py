from data_loader import Data_Loader, Data_Loader_B, DatasetTypes
from parameter import *
from tester import Tester
from torch.backends import cudnn
from trainer import Trainer
from utils import make_folder


def main(config):
    # For fast training
    cudnn.benchmark = True

    if config.train:

        # Create directories if not exist
        make_folder(config.model_save_path, config.version)
        make_folder(config.sample_path, config.version)
        make_folder(config.log_path, config.version)

        if config.version.lower() == "parsenet":
            data_loader = Data_Loader(
                config.img_path,
                config.label_path,
                config.imsize,
                config.batch_size,
                config.train,
            )
        elif config.version.lower() == "blurd3d":
            data_loader = Data_Loader_B(
                config.img_path,
                config.imsize,
                config.batch_size,
                dataset_type=DatasetTypes.BLURD3D,
            )
        elif config.version.lower() == "blurdsd":
            data_loader = Data_Loader_B(
                config.img_path,
                config.imsize,
                config.batch_size,
                dataset_type=DatasetTypes.BLURDSD,
            )
        elif config.version.lower() == "blurdboth":
            data_loader = Data_Loader_B(
                config.img_path,
                config.imsize,
                config.batch_size,
                dataset_type=DatasetTypes.BLURDBOTH,
            )
        else:
            raise ValueError()

        val_loader = Data_Loader(
            config.val_img_path,
            config.val_label_path,
            config.imsize,
            config.batch_size,
            config.train,
        )
        trainer = Trainer(data_loader.loader(), val_loader.loader(), config=config)
        trainer.train()
    else:
        tester = Tester(config)
        tester.test()


if __name__ == "__main__":
    config = get_parameters()
    print(config)
    main(config)
