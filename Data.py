import torchvision.datasets as dset
import torchvision.transforms as transforms
import baseline_model
from torch.utils.data import DataLoader, random_split
from torchtext.vocab import Vocab
from itertools import chain
from collections import Counter
cap = dset.CocoCaptions(root = 'coco/images',
                        annFile = 'coco/annotations/captions_val2017.json',
                        transform=transforms.ToTensor())

def unique_words(lines):
    return set(chain(*(line.split(' ') for line in lines if line)))



total_length = len(cap)
train_length = int(0.8 * total_length)
val_length = int(0.1 * total_length)
test_length = int(0.1 * total_length)

batch_size = 64

train_dataset, val_dataset, test_dataset = random_split(cap, [ train_length, val_length, test_length ])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

# embed_size = 128
# words = unique_words(target)
# vocab_size = len(words)
# vocab = Vocab(Counter(words))

print(words)

#enc_model = baseline_model.EncoderModel(128)


# for i in range(num_epochs):
#
#     zero = torch.FloatTensor([0.0])
#     sub, pred, obj = torch.split(spo, [sub_dim, rel_dim, obj_dim], dim=1)
#     neg_sub, neg_pred, neg_obj = torch.split(sno, [sub_dim, rel_dim, obj_dim], dim=1)
#     criterion = lambda pos, neg : torch.sum(torch.max(Variable(zero).to(device), 1 - pos + neg))
#     optimizer.zero_grad()
#
#     # pos_score = model(Variable(sub).to(device), Variable(pred).to(device), Variable(obj).to(device), batch_size)
#     # neg_score = model(Variable(neg_sub).to(device), Variable(neg_pred).to(device), Variable(neg_obj).to(device), batch_size)
#     # loss = criterion(pos_score, neg_score)
#     # loss.backward()
#     # optimizer.step()
#     #
#     # return loss.item()
#     #
#
#
#
#
#
