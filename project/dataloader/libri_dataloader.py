
import torch
import torch.nn as nn
from torch.nn.modules import transformer
import torchaudio
import torch.nn.functional as F
import pytorch_lightning as pl

from torch.utils.data import Dataset
from torch.utils.data import random_split
from torch.utils.data import DataLoader


class EnglishVocab:
    def __init__(self, vocab_path="project/dataloader/vocab.txt") -> None:
        self.vocab_path = vocab_path

        vocab_list = [line.rstrip('\n') for line in open(vocab_path)]

        self.vocab_dict = {}
        for i, item in enumerate(vocab_list):
            self.vocab_dict[item.lower()] = i

        self.inverted_vocab_dict = {value: key for (
            key, value) in self.vocab_dict.items()}

        self.SOS_TOKEN = self.stoi("<SOS>")
        self.EOS_TOKEN = self.stoi("<EOS>")
        self.PAD_TOKEN = self.stoi("<PAD>")

    def stoi(self, s: str):
        s = s.lower()
        return self.vocab_dict[s]

    def itos(self, i: int):
        return self.inverted_vocab_dict[i]


class MelSpectrogramDataset(Dataset):
    def __init__(self, dataset, transform, vocab: EnglishVocab, has_transcript=True) -> None:
        super(MelSpectrogramDataset).__init__()

        self.max_len = 100

        self.dataset = dataset
        self.transform = transform
        self.has_transcript = has_transcript
        self.vocab = vocab

    def parse_audio(self, wave):
        wave = self.transform(wave)
        return wave

    def parse_text(self, text):
        text = text.lower()
        text = text.replace(" ", "_")

        int_sequence = []
        for c in text:
            ch = self.vocab.stoi(c)
            int_sequence.append(ch)

        int_seq = [self.vocab.SOS_TOKEN] + \
            int_sequence + [self.vocab.EOS_TOKEN]
        # int_seq = int_sequence
        return int_seq, text

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index):

        data = self.dataset[index]
        wave = data[0]
        spect = self.parse_audio(wave)

        if self.has_transcript:
            # TODO: filter text (all charactor should exist in vocab)
            text = data[2]
            # LIMIT max length
            text = text[:self.max_len]
            int_seq, text = self.parse_text(text)
            int_seq = int_seq[:self.max_len]
            # print(len(int_seq))
            return spect, int_seq, text
        else:
            return spect


class LibriSpeechDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, download_path, train_url, test_url):
        super().__init__()

        self.max_len = 100

        self.batch_size = batch_size
        self.download_path = download_path
        self.train_url = train_url
        self.test_url = test_url

        self.vocab = EnglishVocab()

        self.train_transforms = nn.Sequential(
            torchaudio.transforms.MelSpectrogram(
                sample_rate=16000, n_mels=64),
            # torchaudio.transforms.FrequencyMasking(freq_mask_param=30),
            # torchaudio.transforms.TimeMasking(time_mask_param=100)
        )

        self.val_transforms = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000, n_mels=64)

        self.test_transforms = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000, n_mels=64)

    def prepare_data(self):
        torchaudio.datasets.LIBRISPEECH(
            self.download_path, url=self.train_url, download=True)

        torchaudio.datasets.LIBRISPEECH(
            self.download_path, url=self.train_url, download=True)

    def setup(self, stage=None):
        full_dataset = torchaudio.datasets.LIBRISPEECH(
            self.download_path, url=self.train_url)

        self.test_dataset = torchaudio.datasets.LIBRISPEECH(
            self.download_path, url=self.train_url)

        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size

        self.train_dataset, self.val_dataset = random_split(
            full_dataset, [train_size, val_size], generator=torch.Generator())

        self.train_dataset = MelSpectrogramDataset(
            self.train_dataset, self.train_transforms, self.vocab)
        self.val_dataset = MelSpectrogramDataset(
            self.val_dataset, self.val_transforms, self.vocab)
        self.test_dataset = MelSpectrogramDataset(
            self.test_dataset, self.test_transforms, self.vocab)

    def pad_labels(self, labels, pad_token, max_len):
        input_len = len(labels)
        if input_len < max_len:
            pad_len = max_len-input_len
            pad_seq = torch.fill_(torch.zeros(pad_len), pad_token).long()
            labels = torch.cat((labels, pad_seq))
        return labels

    def collate_fn(self, data):
        # data : 1 batch
        # input = spects, label = int_seq
        # it's real length (before padding)
        input_lengths, label_lengths = [], []
        # texts is not used, it exist just for debugging
        spects, int_seqs, texts = [], [], []
        for spect, int_seq, text in data:

            spect = spect.squeeze(0).transpose(0, 1)
            spects.append(spect)
            input_lengths.append(spect.shape[0]//2)

            int_seq = torch.LongTensor(int_seq)
            int_seq = self.pad_labels(
                int_seq, self.vocab.PAD_TOKEN, self.max_len)
            int_seqs.append(int_seq)
            label_lengths.append(len(int_seq))

            texts.append(text)

        # for se in spects:
        #     print(se.size())
        spects = nn.utils.rnn.pad_sequence(
            spects, batch_first=True).transpose(1, 2)  # .unsqueeze(1).transpose(2, 3)

        # Not just padding int_seqs to max length in int_seqs,
        # We have to pad it to self.max_len (300)
        int_seqs = nn.utils.rnn.pad_sequence(int_seqs, batch_first=True)

        # int_seqs = torch.Tensor(int_seqs)
        # int_seqs = F.pad(int_seqs, pad=(0, 2), mode="constant",
        #                  value=self.vocab.PAD_TOKEN)

        # int_seqs = nn.utils.rnn.pad_sequence(
        #     int_seqs, batch_first=True, padding_value=self.vocab.PAD_TOKEN)
        # texts = nn.utils.rnn.pad_sequence(texts, batch_first=True)

        input_lengths = torch.LongTensor(input_lengths)
        label_lengths = torch.LongTensor(label_lengths)

        return spects, int_seqs, input_lengths, label_lengths

    def train_dataloader(self):
        return DataLoader(dataset=self.train_dataset,
                          batch_size=self.batch_size,
                          collate_fn=lambda x: self.collate_fn(x),
                          shuffle=True)

    def val_dataloader(self):
        return DataLoader(dataset=self.val_dataset,
                          batch_size=self.batch_size,
                          collate_fn=lambda x: self.collate_fn(x))

    def test_dataloader(self):
        return DataLoader(dataset=self.test_dataset,
                          batch_size=self.batch_size,
                          collate_fn=lambda x: self.collate_fn(x))


if __name__ == "__main__":

    # vocab = EnglishVocab()

    train_url = "train-clean-100"
    test_url = "test-clean"
    dm = LibriSpeechDataModule(8, './data', train_url, test_url)

    dm.setup()

    train_loader = dm.train_dataloader()

    for spects, int_seqs, input_lengths, label_lengths in train_loader:

        print(spects.size())
        print(int_seqs.size())
