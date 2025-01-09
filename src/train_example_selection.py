import os
import sys
from argparse import ArgumentParser

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

# from pytorch_lightning.utilities.seed import seed_everything  # This is already imported correctly
from pytorch_lightning.strategies.ddp import DDPStrategy

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from example_selection import ExampleSelection


if __name__ == "__main__":
    parser = ArgumentParser()

    # Add arguments for the Trainer manually
    parser.add_argument("--save_top_k", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--project_name", type=str, default="test")
    parser.add_argument("--run_name", type=str, default="test")
    parser.add_argument("--entity", type=str, default="test")
    parser.add_argument("--enable_wandb", action='store_true')

    # Add custom model-specific arguments using the ExampleSelection class
    parser = ExampleSelection.add_model_specific_args(parser)
    
    # Manually add Trainer-related arguments that were previously handled by `add_argparse_args`
    parser.add_argument("--max_epochs", type=int, default=10)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--num_sanity_val_steps", type=int, default=0)
    parser.add_argument("--val_check_interval", type=float, default=1.0)
    
    # Parse the arguments
    args, _ = parser.parse_known_args()

    # Set the random seed for reproducibility
    if args.seed is not None:
        seed_everything(seed=args.seed)

    # Set up Wandb logging if enabled
    logger = WandbLogger(project=args.project_name, name=args.run_name, entity=args.entity) if args.enable_wandb else None

    # Define the callbacks
    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        # Optionally add a ModelCheckpoint callback if needed
        # ModelCheckpoint(monitor="val_loss", save_top_k=args.save_top_k),
    ]

    # Initialize the Trainer object
    trainer = Trainer(
        logger=logger if logger else True,
        callbacks=callbacks,
        max_epochs=args.max_epochs,
        num_sanity_val_steps=args.num_sanity_val_steps,
        val_check_interval=args.val_check_interval
    )

    # Initialize the model
    model = ExampleSelection(**vars(args))

    # Start the training process
    trainer.fit(model)
