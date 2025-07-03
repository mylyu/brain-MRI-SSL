import argparse
import os
import sys
import torch
import omegaconf
import pytorch_lightning as pl
import torchvision.models as models
import torch
import torch.nn as nn
import torch.nn.functional as F
import traceback
import random
import numpy as np
from typing import Any, Callable, Dict, List, Tuple, Union
from omegaconf import OmegaConf, DictConfig
from solo.data.classification_dataloader import prepare_data
from solo.utils.metrics import accuracy_at_k, weighted_mean
from torchmetrics import AUROC, Precision, Recall

class LinearModel(pl.LightningModule):
    def __init__(self, backbone: nn.Module, cfg: omegaconf.DictConfig, loss_func: Callable = None, **kwargs):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.backbone = backbone
        self.loss_func = loss_func if loss_func is not None else nn.CrossEntropyLoss()
        if hasattr(self.backbone, "inplanes"): 
            features_dim = self.backbone.inplanes
        else: 
            features_dim = self.backbone.num_features
        
        # Dynamically create the classifier head
        self.classifier = nn.Linear(features_dim, self.hparams.model.num_classes)  
        if not self.hparams.get("finetune", False):
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Initialize metrics based on the downstream task's actual number of classes
        task_type = "multiclass"
        self.precision_metric = Precision(task=task_type, num_classes=self.hparams.data.num_classes, average='macro')
        self.recall_metric = Recall(task=task_type, num_classes=self.hparams.data.num_classes, average='macro')
        self.validation_step_outputs = []

    def forward(self, x: torch.tensor):
        feats = self.backbone(x)
        logits = self.classifier(feats)   # The output dimension matches the model's head
        return {"logits": logits}

    def validation_step(self, batch, batch_idx): 
        x, target = batch
        out = self(x)
        
        # `model_logits` has the dimension of the model's head (e.g., [batch_size, 100])
        model_logits = out["logits"]

        # `task_num_classes` is the actual number of classes in the downstream task (e.g., 4)
        task_num_classes = self.hparams.data.num_classes
        
        # Slice the logits to match the downstream task dimension
        task_logits = model_logits[:, :task_num_classes]
        
        # The loss is calculated between the sliced logits and the targets.
        loss = self.loss_func(task_logits, target)

        # Calculate top-k accuracy. `accuracy_at_k` can handle mismatched dimensions if `target` is valid.
        k = min(5, task_num_classes)
        top_k = (1, k) if k > 1 else (1,)
        accuracies = accuracy_at_k(task_logits, target, top_k=top_k)
        
        acc1 = accuracies[0]
        acc5 = accuracies[1] if len(accuracies) > 1 else torch.tensor(0.0)

        # Predictions for other metrics must be based on the sliced `task_logits`.
        preds = torch.argmax(task_logits, dim=1)
        self.precision_metric.update(preds, target)
        self.recall_metric.update(preds, target)
        self.validation_step_outputs.append({
            "val_loss": loss.detach(),
            "val_acc1": acc1.detach(),
            "val_acc5": acc5.detach(),
            "batch_size": x.size(0)
        })

    def on_validation_epoch_end(self):
        """Aggregates and logs metrics at the end of the validation epoch."""
        if not self.validation_step_outputs: return
        val_acc1 = weighted_mean(self.validation_step_outputs, "val_acc1", "batch_size")
        precision = self.precision_metric.compute()
        recall = self.recall_metric.compute()
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        
        print("\n" + "="*20 + " Validation Results " + "="*20)
        print(f"  - Accuracy (Acc1): \t{val_acc1.item():.2f}%")
        print(f"  - Precision:       \t{precision.item() * 100:.2f}%")
        print(f"  - Recall:          \t{recall.item() * 100:.2f}%")
        print(f"  - F1 Score:        \t{f1.item() * 100:.2f}%")
        print("="*60 + "\n")

        self.validation_step_outputs.clear()
        self.precision_metric.reset()
        self.recall_metric.reset()

    def training_step(self, batch, batch_idx): pass
    def configure_optimizers(self): return None

def seed_everything(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

SEED = 42 
seed_everything(SEED)

# Default base directory for datasets, assuming it's in the project root.
# Users running this script should place the standard datasets here.
BASE_DATA_DIR = "../datasets"

def run_reproduce(cfg: DictConfig, ckpt_path: str):
    """
    Initializes the model, data loader, and trainer, then starts the validation process.
    
    Args:
        cfg (DictConfig): An OmegaConf object containing all necessary configurations.
        ckpt_path (str): The full path to the model checkpoint to be validated.
    """

    # Dynamically create the backbone based on the configuration
    if cfg.model.backbone == 'resnet18':
        backbone = models.resnet18(weights=None)
    else:  
        backbone = models.resnet50(weights=None)  # default to resnet50
    backbone.fc = nn.Identity()                      
    model = LinearModel(backbone=backbone, cfg=cfg)
     
    # Prepare the validation dataloader
    try:
        _, val_loader = prepare_data("custom", 
            train_data_path=os.path.join(cfg.data.dataset_path, 'train'), 
            val_data_path=os.path.join(cfg.data.dataset_path, 'val'),
            batch_size=cfg.batch_size,
            num_workers=4,
        )
    except Exception as e:
        print(f"\nError: Failed to prepare data. Please check the dataset path: {cfg.data.dataset_path}\n{e}")
        return
    
    # Initialize the Pytorch Lightning Trainer and start the validation process
    trainer = pl.Trainer(accelerator="gpu", devices=[0], precision='16-mixed', logger=False, enable_progress_bar=True)
    
    print(f"--- Validating checkpoint: {ckpt_path} ---")
    try:
        trainer.validate(model, ckpt_path=ckpt_path, dataloaders=val_loader, verbose=False)
    except Exception as e:
        print("\n" + "!"*20 + " A critical error occurred during validation " + "!"*20)
        traceback.print_exc()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=['4class', '16class', 'custom'])
    parser.add_argument("--ckpt", required=True, type=str)                                
    parser.add_argument("--backbone", default='resnet50', choices=['resnet18', 'resnet50']) 
    parser.add_argument("--custom_task_classes", type=int)
    parser.add_argument("--custom_dataset_path", type=str)
    args = parser.parse_args()
    ckpt_full_path = args.ckpt

    # Configure task-specific parameters based on the selected mode
    if args.task == '4class':
        num_classes = 4
        dataset_path = os.path.join(BASE_DATA_DIR, "brain_class_4")
    elif args.task == '16class':
        num_classes = 16
        dataset_path = os.path.join(BASE_DATA_DIR, "brain_class_16")
    else:
        if not (args.custom_task_classes and args.custom_dataset_path):
            parser.error("For '--task custom', you must provide both '--custom_task_classes' and '--custom_dataset_path'.")
        num_classes = args.custom_task_classes
        dataset_path = args.custom_dataset_path

    if not os.path.exists(ckpt_full_path):
        parser.error(f"Checkpoint file not found: {ckpt_full_path}\n")
    if not os.path.exists(dataset_path):
        parser.error(f"Dataset path not found: {dataset_path}\n")

    # Get the model's true classifier dimension
    try:    
        state_dict = torch.load(ckpt_full_path, map_location="cpu")['state_dict']
        if 'classifier.weight' in state_dict:
            model_num_classes_from_ckpt = state_dict['classifier.weight'].shape[0]
        else:
            raise ValueError(f"Could not find 'classifier.weight' in the provided checkpoint!")
    except Exception as e:
        parser.error(f"Failed to load or parse checkpoint file: {e}")
    
    # Create the final configuration object
    cfg = OmegaConf.create({
        "model": { "num_classes": model_num_classes_from_ckpt, "backbone": args.backbone },  # This dimension tells LinearModel how to initialize itself to match the checkpoint
        "data": { "num_classes": num_classes, "dataset_path": dataset_path },                # This dimension tells LinearModel how to evaluate (e.g., slicing logits)
        "batch_size": 128,   
        "finetune": False   
    })
    
    run_reproduce(cfg, ckpt_full_path)

if __name__ == "__main__":
    main()
