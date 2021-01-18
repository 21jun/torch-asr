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


dm = LibriSpeechDataModule(8, './data', train_url, test_url)


dm.setup()

train_loader = dm.train_dataloader()
train_ds = dm.train_dataset

index_map = dm.train_dataset.index_map

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
# criterion = nn.CTCLoss(blank=1).to(device)
criterion = nn.CrossEntropyLoss(reduction='mean').to(device)
# criterion = nn.CrossEntropyLoss(reduction='mean').to(device)
scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=hparams['learning_rate'],
                                          steps_per_epoch=int(
                                              len(train_loader)),
                                          epochs=hparams['epochs'],
                                          anneal_strategy='linear')


def decoder(output, labels, label_lengths, blank_label=1):
    output = output.transpose(0, 1)
    print(output.size())
    # print(labels)

    for sent in labels:
        sentence = []
        for char in sent:
            char = char.item()
            sentence.append(index_map[char])

        print("".join(sentence))

    arg_maxes = torch.argmax(output, dim=2)
    print(arg_maxes.size())
    sentences = []
    indexeses = []
    for seq in arg_maxes:
        sentence = []
        indexs = []
        for index in seq:
            index = index.item()
            char = index_map[index]
            sentence.append(char)
            indexs.append(index)
        sentences.append(''.join(sentence))
        indexeses.append(indexs)
    print(sentences)
    # print(indexeses)
    # print(len(sentences))
    # print(len(sentences[0]))
    return


for batch_idx, _data in enumerate(train_loader):

    optimizer.zero_grad()

    spectrograms, labels, input_lengths, label_lengths = _data
    spectrograms, labels = spectrograms.to(device), labels.to(device)

    output = model(spectrograms)
    # output = F.log_softmax(output, dim=2)
    # output = output.transpose(0, 1)
    # labels = labels.argmax(1)
    # print(labels.unsqueeze(1))
    print(labels)
    # if batch_idx % 20 == 0:
    #     decoder(output, labels, label_lengths)

    # loss = criterion(output, labels, input_lengths, label_lengths)

    print(output.size())
    print(labels.size())

    loss = criterion(output, labels)
    # loss = criterion(output, labels)
    loss.backward()

    optimizer.step()
    scheduler.step()

    print(loss, batch_idx)
