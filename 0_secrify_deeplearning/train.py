from model import SecrifyDataset, LightningModule
from torch.utils.data import DataLoader
import sys
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint

data_directory = sys.argv[1]
dataset = sys.argv[2]

# For results
train_files = [f'{data_directory}/{dataset}_train.csv'] 
valid_files = [f'{data_directory}/{dataset}_valid.csv']
test_files = [f'{data_directory}/{dataset}_test.csv'] 

batch_size = 128

train_set = SecrifyDataset(dataset_locs=train_files)
valid_set = SecrifyDataset(dataset_locs=valid_files)
test_set = SecrifyDataset(dataset_locs=test_files)
collator = train_set.collate_fn
train_loader = DataLoader(train_set, batch_size, shuffle=True, pin_memory=True, collate_fn=collator)
valid_loader = DataLoader(valid_set, batch_size, shuffle=False, pin_memory=True, collate_fn=collator)
test_loader = DataLoader(test_set, batch_size, shuffle=False, pin_memory=True, collate_fn=collator)

lightning_module = LightningModule(dataset_name=dataset,steps_per_training_epoch=len(train_loader))

early_stopping = EarlyStopping(monitor='validation_loss',patience=15,mode='min')
lr_monitor = LearningRateMonitor(logging_interval='step')
callbacks = [early_stopping, lr_monitor]
model_dir = 'saved_models/'
model_checkpointing = ModelCheckpoint(
    dirpath=model_dir,
    save_top_k=1,
    verbose=True,
    monitor='validation_loss',
    mode='min',
    filename=f'model_{dataset}',
)

callbacks.append(model_checkpointing)
trainer = pl.Trainer(
    accelerator='gpu',
    max_epochs=15,
    callbacks=callbacks,
    default_root_dir=model_dir,
    val_check_interval=0.1,
)

trainer.fit(lightning_module, train_loader, valid_loader)

# (7) Evaluate predictions on the test set
trainer.test(dataloaders=test_loader, ckpt_path='best')