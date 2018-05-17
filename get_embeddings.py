
#Code to test the encoder.py
import numpy as np
import torch
from torch.autograd import Variable
from collections import Counter
import torch.utils.data as data_utils
from keras.datasets import imdb
from keras.preprocessing.sequence import pad_sequences
from encoder import PositionalEncoder

#Hyper params
MAXLEN = 100
BATCH_SIZE=1
EMB_DIM = 50

#Load the imdb keras data
train_set,test_set = imdb.load_data(num_words=1000, index_from=3)
x_train,y_train = train_set[0],train_set[1]
x_test,y_test = test_set[0],test_set[1]
word_to_id = imdb.get_word_index()
word_to_id = {k:(v+3) for k,v in word_to_id.items()}
word_to_id["<PAD>"] = 0
word_to_id["<START>"] = 1
word_to_id["<UNK>"] = 2
x_train_pad = pad_sequences(x_train,maxlen=MAXLEN)

#Create batches
train_data = data_utils.TensorDataset(torch.from_numpy(x_train_pad).type(torch.LongTensor),torch.from_numpy(y_train).type(torch.DoubleTensor))
train_loader = data_utils.DataLoader(train_data,batch_size=BATCH_SIZE,drop_last=True)

#Initialize the model
pe = PositionalEncoder(len(word_to_id),EMB_DIM,MAXLEN,BATCH_SIZE)

#Retrieve the embeddings
for batch_idx,train in enumerate(train_loader):
    x,y = Variable(train[0]),Variable(train[0])
    print(pe(x))