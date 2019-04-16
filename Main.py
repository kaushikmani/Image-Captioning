import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch
import torch.nn as nn
from Model import EncoderModel, DecoderModel
import numpy as np
from torch.utils.data import DataLoader, random_split
from torchtext.vocab import Vocab
from itertools import chain
import nltk
from collections import Counter
from pycocotools.coco import COCO
from DataLoader import CocoDataset
from torch.nn.utils.rnn import pack_padded_sequence

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Main():
    def __init__(self):

        self.json = 'coco/annotations/captions_val2017.json'
        self.minCount = 5
        self.imgDir = 'coco/images'
        self.batch_size = 64
        self.embed_size = 128
        self.hidden_size = 128
        self.num_layers = 1
        self.learning_rate = 0.001
        self.criterion = nn.CrossEntropyLoss()
        self.num_epochs = 10
        self.vocab = self.build_vocab(self.json, self.minCount)
        self.transform = self.transform()
        self.vocab_size = len(self.vocab)

        self.enc_model = EncoderModel(self.embed_size).to(device)
        self.dec_model = DecoderModel(self.embed_size , self.hidden_size, self.vocab_size, self.num_layers).to(device)

        self.params = list(self.dec_model.parameters()) + list(self.enc_model.linear.parameters()) + list(self.enc_model.bn.parameters())
        self.optimizer = torch.optim.Adam(self.params, lr=self.learning_rate)

    def build_vocab(self, json, threshold):
        coco = COCO(json)
        counter = Counter()
        ids = coco.anns.keys()

        for i, id in enumerate(ids):
            caption = str(coco.anns[id]['caption'])
            tokenized_caption = nltk.tokenize.word_tokenize((caption.lower()))
            counter.update(tokenized_caption)


        words = [word for word, count in counter.items() if count >= threshold]

        vocab = Vocab(Counter(words), specials=['<pad>', '<start>', '<end>', '<unk>'])

        return vocab

    def transform(self):
        transform = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))
        ])

        return transform

    def collate_fn(self, data):
        """Creates mini-batch tensors from the list of tuples (image, caption).
        """

        # Sort a data list by caption length (descending order).
        data.sort(key=lambda x: len(x[1]), reverse=True)
        images, captions = zip(*data)

        # Merge images (from tuple of 3D tensor to 4D tensor).
        images = torch.stack(images, 0)

        # Merge captions (from tuple of 1D tensor to 2D tensor).
        lengths = [len(cap) for cap in captions]
        targets = torch.zeros(len(captions), max(lengths)).long()

        for i, cap in enumerate(captions):
            end = lengths[i]
            targets[i, :end] = cap[:end]

        return images, targets, lengths


    def train(self):

        coco = CocoDataset(root=self.imgDir, json=self.json, vocab=self.vocab, transform=self.transform)

        total_length = len(coco)
        train_length = int(0.8 * total_length)
        val_length = int(0.1 * total_length)
        test_length = int(0.1 * total_length)

        train_dataset, val_dataset, test_dataset = random_split(coco, [train_length, val_length, test_length])

        train_data_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True,collate_fn=self.collate_fn)



        total_step = len(train_data_loader)

        for epoch in range(self.num_epochs):
            for i, (images, captions, lengths) in enumerate(train_data_loader):

                images = images.to(device)
                captions = captions.to(device)

                features = self.enc_model(images)
                outputs = self.dec_model(features, captions, lengths)

                labels = pack_padded_sequence(captions, lengths, batch_first=0)[0]

                loss = self.criterion(outputs, labels)
                self.enc_model.zero_grad()
                self.dec_model.zero_grad()

                loss.backward()
                self.optimizer.step()

                if i%50 == 0:
                    val_loss = self.validate(val_dataset)
                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Validation Loss: {:.4f}, Perplexity: {:5.4f}'.format(epoch+1, self.num_epochs, i+1, total_step, loss.item(), val_loss, np.exp(val_loss)))


    def validate(self, val_dataset):

        val_data_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=self.batch_size, shuffle=True,collate_fn=self.collate_fn)


        total_steps = len(val_data_loader)

        self.enc_model.eval()
        self.dec_model.eval()
        val_loss = 0.0
        for i, (images, captions, lengths) in enumerate(val_data_loader):
            images = images.to(device)
            captions = captions.to(device)

            features = self.enc_model(images)
            outputs = self.dec_model(features, captions, lengths)

            labels = pack_padded_sequence(captions, lengths, batch_first=0)[0]

            val_loss += self.criterion(outputs, labels).item()

        return (val_loss/total_steps)

if __name__ == '__main__':

    main = Main()
    main.train()

