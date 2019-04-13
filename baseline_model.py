import torch.nn as nn
import torch
import torchvision.models as models
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence


class EncoderModel(nn.Module):

    def __init__(self, embed_size):
        super(EncoderModel, self).__init__()

        #Using pretrained resnet model and removing the last layer
        resnet = models.resnet18(pretrained=True)
        modules = list(resnet.children())[:-1]

        #Using that resnet model
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
        self.init_weights()

    def init_weights(self):

        self.linear.weight.data.normal_(0.0, 0.02)
        self.linear.bias.data.fill_(0)

    def forward(self, images):

        features = self.resnet(images)
        features = Variable(features.data)
        features = features.view(features.size(0), -1)
        features = self.bn(self.linear(features))
        return features


class DecoderModel(nn.Module):

    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, biDirectional=False):
        super(DecoderModel, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True, bidirectional=biDirectional)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.init_weights()


    def init_weights(self):

        #Initial embedding - Can try Word2Vec or even advanced embedding methods.
        self.embed.weight.data.uniform_(-0.1, 0.1)
        self.linear.weight.data.uniform_(-0.1, 0.1)
        self.linear.bias.data.fill_(0)

    def forward(self, features, captions, lengths):
        
        embedding = self.embed(captions)
        embedding = torch.cat((features.unsqueeze(1), embedding), 1)
        packed = pack_padded_sequence(embedding, lengths, batch_first=True)
        hidden, _ = self.lstm(packed)
        output = self.linear(hidden[0])
        return output

