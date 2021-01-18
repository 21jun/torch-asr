import torch
import torch.nn as nn
from torch.nn.modules import transformer
import torchaudio
import pytorch_lightning as pl

from torch.utils.data import Dataset
from torch.utils.data import random_split
from torch.utils.data import DataLoader

SOS_token = 29
EOS_token = 30
PAD_token = 0


class MelSpectrogramDataset(Dataset):
    def __init__(self, dataset, transform, has_transcript=True) -> None:
        super(MelSpectrogramDataset).__init__()

        self.eos_id = EOS_token
        self.sos_id = SOS_token

        self.dataset = dataset
        self.transform = transform
        self.has_transcript = has_transcript

        char_map_str = """
        ' 0
        <SPACE> 1
        a 2
        b 3
        c 4
        d 5
        e 6
        f 7
        g 8
        h 9
        i 10
        j 11
        k 12
        l 13
        m 14
        n 15
        o 16
        p 17
        q 18
        r 19
        s 20
        t 21
        u 22
        v 23
        w 24
        x 25
        y 26
        z 27
        """
        self.char_map = {}
        self.index_map = {}
        for line in char_map_str.strip().split('\n'):
            ch, index = line.split()
            self.char_map[ch] = int(index)
            self.index_map[int(index)] = ch
        self.index_map[1] = ' '

    def parse_audio(self, wave):
        wave = self.transform(wave)
        return wave

    def parse_text(self, text):
        text = text.lower()
        int_sequence = []
        for c in text:
            if c == ' ':
                ch = self.char_map['<SPACE>']
            else:
                ch = self.char_map[c]
            int_sequence.append(ch)

        # int_seq = [self.sos_id] + int_sequence + [self.eos_id]
        int_seq = int_sequence
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
            int_seq, text = self.parse_text(text)
            return spect, int_seq, text
        else:
            return spect


class LibriSpeechDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, download_path, train_url, test_url):
        super().__init__()
        self.batch_size = batch_size
        self.download_path = download_path
        self.train_url = train_url
        self.test_url = test_url

        self.train_transforms = nn.Sequential(
            torchaudio.transforms.MelSpectrogram(
                sample_rate=16000, n_mels=128),
            # torchaudio.transforms.FrequencyMasking(freq_mask_param=30),
            # torchaudio.transforms.TimeMasking(time_mask_param=100)
        )

        self.val_transforms = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000, n_mels=128)

        self.test_transforms = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000, n_mels=128)

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
            self.train_dataset, self.train_transforms)
        self.val_dataset = MelSpectrogramDataset(
            self.val_dataset, self.val_transforms)
        self.test_dataset = MelSpectrogramDataset(
            self.test_dataset, self.test_transforms)

    def collate_fn(data):
        # input = spects, label = int_seq
        # it's real length (before padding)
        input_lengths, label_lengths = [], []
        # texts is not used, it exist just for debugging
        spects, int_seqs, texts = [], [], []
        for spect, int_seq, text in data:

            spect = spect.squeeze(0).transpose(0, 1)
            spects.append(spect)
            input_lengths.append(spect.shape[0]//2)

            int_seq = torch.Tensor(int_seq)
            int_seqs.append(int_seq)
            label_lengths.append(len(int_seq))

            texts.append(text)

        # for se in spects:
        #     print(se.size())
        spects = nn.utils.rnn.pad_sequence(
            spects, batch_first=True).unsqueeze(1).transpose(2, 3)  # .unsqueeze(1).transpose(2, 3)

        int_seqs = nn.utils.rnn.pad_sequence(
            int_seqs, batch_first=True)
        # texts = nn.utils.rnn.pad_sequence(texts, batch_first=True)
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
