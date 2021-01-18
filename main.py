import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
from project.dataloader.librispeech_dataloader import LibriSpeechDataModule
from project.model.cnn_model import SpeechRecognitionModel, SpeechRecognitionModule
import pytorch_lightning as pl


train_url = "train-clean-100"
test_url = "test-clean"
hparams = {
    "n_cnn_layers": 3,
    "n_rnn_layers": 5,
    "rnn_dim": 512,
    "n_class": 29,
    "n_feats": 128,
    "stride": 2,
    "dropout": 0.1,
    "learning_rate": 5e-4,
    "batch_size": 16,
    "epochs": 1
}


dm = LibriSpeechDataModule(16, './data', train_url, test_url)
model = SpeechRecognitionModule(n_cnn_layers=hparams['n_cnn_layers'], n_rnn_layers=hparams['n_rnn_layers'], rnn_dim=hparams['rnn_dim'],
                                n_class=hparams['n_class'], n_feats=hparams['n_feats'], stride=hparams['stride'], dropout=hparams['dropout'], len_train=2000)

trainer = pl.Trainer(max_epochs=1, gpus=1)
trainer.fit(model, dm)
