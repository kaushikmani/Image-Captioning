import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch
import baseline_model
from torch.utils.data import DataLoader, random_split
from torchtext.vocab import Vocab
from itertools import chain
import nltk
from collections import Counter

transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

cap = dset.CocoCaptions(root = 'coco/images',
                        annFile = 'coco/annotations/captions_val2017.json',
                        transform=transform)

img , target = cap[2];
print(img.size())
print(target)
finalTarget = []

# for img,target in cap:
#     finalTarget = finalTarget + target
# print(len(finalTarget))
# print(finalTarget)
k = 0
result = set()
for img, target in cap:
    for t in target:
        # result.add(set(nltk.tokenize.word_tokenize(str(t).lower())))
        result = set(chain(result,set(nltk.tokenize.word_tokenize(str(t).lower()))))
    # print(target)
    # k = k + 1
    # if k == 10:
    #     break

result = list(result)
print(result)

def unique_words(lines):
    return set(chain(*(line.split(' ') for line in lines if line)))



total_length = len(cap)
train_length = int(0.8 * total_length)
val_length = int(0.1 * total_length)
test_length = int(0.1 * total_length)

batch_size = 64

train_dataset, val_dataset, test_dataset = random_split(cap, [ train_length, val_length, test_length ])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

embed_size = 128
hidden_size = 128
num_layers = 1
words = len(result)
vocab_size = len(words)
vocab = Vocab(Counter(words))

#print(words)

enc_model = baseline_model.EncoderModel(embed_size)
dec_model = baseline_model.DecoderModel(embed_size , hidden_size, vocab_size,num_layers)

num_epochs = 1

for i in range(num_epochs):

    for j,(images,)

    # zero = torch.FloatTensor([0.0])
    # sub, pred, obj = torch.split(spo, [sub_dim, rel_dim, obj_dim], dim=1)
    # neg_sub, neg_pred, neg_obj = torch.split(sno, [sub_dim, rel_dim, obj_dim], dim=1)
    # criterion = lambda pos, neg : torch.sum(torch.max(Variable(zero).to(device), 1 - pos + neg))
    # optimizer.zero_grad()

    # pos_score = model(Variable(sub).to(device), Variable(pred).to(device), Variable(obj).to(device), batch_size)
    # neg_score = model(Variable(neg_sub).to(device), Variable(neg_pred).to(device), Variable(neg_obj).to(device), batch_size)
    # loss = criterion(pos_score, neg_score)
    # loss.backward()
    # optimizer.step()
    #
    # return loss.item()
    #





