from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
# from omegaconf import DictConfig
# import hydra
import torch
from model import ChestXRayCNN
from dataset import ClassificationDataModule
import config
import pytorch_lightning as pl
pl.utilities.seed.seed_everything(seed = 1)

def main():
    dm = ClassificationDataModule(config)
    model = ChestXRayCNN(config)

    checkpoint_callback = ModelCheckpoint(
        dirpath=config.checkpoint_dir,
        filename='bs_{0}_lr_{1}'.format(config.batch_size, config.lr),
        verbose=config.verbose
    )

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=15,
        verbose=config.verbose,
        mode='min'
    )
    device_num = 1
    print("each batch size:", config.batch_size // device_num)
    trainer = Trainer(
        callbacks=[checkpoint_callback, early_stopping],
        gpus = device_num, 
        accelerator = "ddp" if device_num > 1 else None,
        # precision=16,
        max_epochs=config.epochs
    )

    trainer.fit(model, dm)
    trainer.save_checkpoint(config.checkpoint_dir+'/Xray.ckpt')
    


if __name__ == "__main__":
    main()