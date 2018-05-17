# Positional-Encoding
Attention is all you need: https://arxiv.org/abs/1706.03762 does not use any recurrent neural networks to encode the sentence. Instead they make use of positional encodings followed by attention. 
In the paper, thay use sine and cosine functions of different frequencies:
```
P E(pos,2i) = sin(pos/10000**(2i/dmodel))
P E(pos,2i+1) = cos(pos/10000**(2i/dmodel))
```
where pos is the position and i is the dimension. That is, each dimension of the positional encoding
corresponds to a sinusoid. The wavelengths form a geometric progression from 2π to 10000 · 2π. We
chose this function because we hypothesized it would allow the model to easily learn to attend by
relative positions, since for any fixed offset k, P Epos+k can be represented as a linear function of
P Epos.

encoder.py provides a class which helps to encode the position/time component along with the word embeddings. Both the position as well as word embeddings are trainiable. Encoding output of this class must be passed through a self attention layer for improved results.

### Syntax
```
import torch
import numpy as np
pe = PositionalEncoder(vocab_size,EMB_DIM,MAXLEN,BATCH_SIZE)
word_seq = Variable(torch.from_numpy(np.array([0,0,34,56,23,1])).type(torch.LongTensor))
pe(word_seq)
```



Its time to throw away LSTM/RNN's . Attention is all you need. 
