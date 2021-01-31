import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
from project.dataloader.libri_dataloader import LibriSpeechDataModule

import pytorch_lightning as pl

from project.model.LAS import Encoder, Decoder, Seq2seq

config_file = "project/model/LAS_config.yaml"
with open(config_file) as f:
    config = yaml.safe_load(f)

train_url = "train-clean-100"
test_url = "test-clean"
use_cuda = True


device = torch.device("cuda" if use_cuda else "cpu")
encoder = Encoder(config['input_feat_dim'],
                  config['enc_hidden_dim'], config['n_layers_enc'])
decoder = Decoder(config['vocab_size'], config['max_out_len'], config['dec_hidden_dim'], config['enc_hidden_dim'],
                  config['sos_id'], config['eos_id'], config['n_layers_dec'], rnn_celltye='gru')
model = Seq2seq(encoder, decoder).to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001,
                       weight_decay=0.0, betas=(0.9, 0.98), eps=1e-9)
criterion = nn.CrossEntropyLoss(reduction='mean').to(device)


dm = LibriSpeechDataModule(16, './data', train_url, test_url)
dm.setup()
train_loader = dm.train_dataloader()
train_ds = dm.train_dataset

vocab = dm.vocab


def calc_cer(predictions, groundtruth, char_dict, print_item=False):
    rev_char_map = dict(map(reversed, char_dict.items()))

    for i in range(predictions.size(0)):

        pred_char_list = []
        org_char_list = []
        org = groundtruth[i].detach().cpu().numpy()
        pred = predictions[i].detach().cpu().numpy()

        for k in pred:
            pred_char_list.append(rev_char_map[k])
        for k in org:
            org_char_list.append(rev_char_map[k])

        pred_text = ''.join(pred_char_list)
        org_text = ''.join(org_char_list)

        pred_text = pred_text.replace("<pad>", "")
        org_text = org_text.replace("<pad>", "")

        if print_item:
            print('{} [Ground truth] {}'.format(i, org_text))
            print('{} [Predicted text] {}'.format(i, pred_text))


for i in range(config['num_epochs']):
    print("Epoch {}".format(i))
    for batch_idx, _data in enumerate(train_loader):

        optimizer.zero_grad()

        spectrograms, labels, input_lengths, label_lengths = _data
        spectrograms, labels = spectrograms.to(device), labels.to(device)

        # print(spectrograms.size())
        # print(labels.size())
        # print(input_lengths)
        # print(label_lengths)

        decoder_outputs, sequence_symbols = model(
            spectrograms, input_lengths, labels)

        # print(decoder_outputs.size())
        # print(labels.size())

        # print(labels)

        loss = criterion(decoder_outputs, labels)
        loss.backward()
        optimizer.step()

        # print(decoder_outputs.size())

        if batch_idx % 50 == 49:
            print(loss.item())
            avg_cer = calc_cer(sequence_symbols, labels,
                               vocab.vocab_dict, print_item=True)
