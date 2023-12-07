import os
import sys
sys.path.append('/apdcephfs/share_733425/vinnylywang/zhanyuwang/Code/xray_chat/')
from pprint import pprint
from configs.config import parser
from dataset.data_module import DataModule
from tools.callbacks import add_callbacks
from models.Xchat import XChatModel
import lightning.pytorch as pl
from lightning.pytorch import seed_everything
from lightning.pytorch.strategies import DeepSpeedStrategy, DDPStrategy


def train(args):
    dm = DataModule(args)
    callbacks = add_callbacks(args)

    if 'deepspeed' in args.strategy:
        strategy = DeepSpeedStrategy(
                stage=3,
                offload_optimizer=True,  # Enable CPU Offloading
                offload_parameters=True
            )
    elif 'ddp' in args.strategy:
        strategy = DDPStrategy(find_unused_parameters=True)
    else:
        strategy = args.strategy


    trainer = pl.Trainer(
        devices=args.devices,
        num_nodes=args.num_nodes,
        strategy=strategy,
        accelerator=args.accelerator,
        precision=args.precision,
        val_check_interval = args.val_check_interval,
        limit_val_batches = args.limit_val_batches,
        max_epochs = args.max_epochs,
        max_steps = args.max_steps,
        log_every_n_steps=args.log_every_n_steps,
        num_sanity_val_steps = args.num_sanity_val_steps,
        accumulate_grad_batches=args.accumulate_grad_batches,
        callbacks=callbacks["callbacks"], 
        logger=callbacks["loggers"]
    )

    if args.ckpt_file is not None:
        model = XChatModel.load_from_checkpoint(args.ckpt_file, strict=False)
    else:
        model = XChatModel(args)
        
    trainer.fit(model, datamodule=dm)


def main():
    args = parser.parse_args()
    os.makedirs(args.savedmodel_path, exist_ok=True)
    pprint(vars(args))
    seed_everything()
    train(args)


if __name__ == '__main__':
    main()