import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
from project.dataloader.librispeech_dataloader import LibriSpeechDataModule
from project.model.cnn_model import SpeechRecognitionModel
import torch.nn.functional as F

train_url = "train-clean-100"
test_url = "test-clean"

dm = LibriSpeechDataModule(8, './data', train_url, test_url)
dm.setup()

train_loader = dm.train_dataloader()
train_ds = dm.train_dataset


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
device = "cuda"
model = SpeechRecognitionModel(
    hparams['n_cnn_layers'], hparams['n_rnn_layers'], hparams['rnn_dim'],
    hparams['n_class'], hparams['n_feats'], hparams['stride'], hparams['dropout']
).to(device)

optimizer = optim.AdamW(model.parameters(), hparams['learning_rate'])
criterion = nn.CTCLoss(blank=1).to(device)
# criterion = nn.CrossEntropyLoss(reduction='mean').to(device)
scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=hparams['learning_rate'],
                                          steps_per_epoch=int(
                                              len(train_loader)),
                                          epochs=hparams['epochs'],
                                          anneal_strategy='linear')


for batch_idx, _data in enumerate(train_loader):

    optimizer.zero_grad()

    spectrograms, labels, input_lengths, label_lengths = _data
    spectrograms, labels = spectrograms.to(device), labels.to(device)

    output = model(spectrograms)
    output = F.log_softmax(output, dim=2)
    output = output.transpose(0, 1)

    # print(output.size())
    # print(labels.size())

    # torch.Size([614, 10, 29])
    # torch.Size([10, 235])

    # print(input_lengths)
    # print(label_lengths)

    loss = criterion(output, labels, input_lengths, label_lengths)
    # loss = criterion(output, labels)
    loss.backward()

    optimizer.step()
    scheduler.step()

    print(loss)
