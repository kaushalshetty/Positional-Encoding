import numpy as np
import torch
from torch.autograd import Variable

def position_encoding_init(n_position, emb_dim):
    ''' Init the sinusoid position encoding table '''

    # keep dim 0 for padding token position encoding zero vector
    position_enc = np.array([
        [pos / np.power(10000, 2 * (j // 2) / emb_dim) for j in range(emb_dim)]
        if pos != 0 else np.zeros(emb_dim) for pos in range(n_position)])
    

    position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2]) # apply sin on 0th,2nd,4th...emb_dim
    position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2]) # apply cos on 1st,3rd,5th...emb_dim
    return torch.from_numpy(position_enc).type(torch.FloatTensor)

class PositionalEncoder(torch.nn.Module):
	"""
	Sets up embedding layer for word sequences as well as for word positions.Both the layers are trainable.
	Returns embeddings of words which also contains the position(time) component
	"""
    def __init__(self,vocab_size,emb_dim,max_len,batch_size):    
		
		"""
		Args:
            vocab_size  : [int] vocabulary size
            emb_dim     : [int] embedding dimension for words
            max_len     : [int] maxlen of input sentence
            batch_size  : [int] batch_size

 
        Returns:
            position encoded word embeddings
 
        Raises:
            nothing
        """    
        super(PositionalEncoder,self).__init__()
        self.batch_size = batch_size
        self.emb_dim = emb_dim
        self.max_len = max_len
        n_position = max_len+1
        self.position_enc = torch.nn.Embedding(n_position, emb_dim, padding_idx=0)
        self.position_enc.weight.data = position_encoding_init(n_position, emb_dim)
        self.src_word_emb = torch.nn.Embedding(vocab_size,emb_dim,padding_idx=0)
        
    def get_absolute_pos(self,word_sequences):
        batch = []
        for word_seq in word_sequences:
            start_idx = 1
            word_pos = []
            for pos in word_seq:
                if int(pos) == 0:
                    word_pos.append(0)
                else:
                    word_pos.append(start_idx)
                    start_idx+=1
            batch.append(torch.from_numpy(np.array(word_pos)).type(torch.LongTensor))
        batch = torch.cat(batch).view(self.batch_size,self.max_len)        
        return Variable(batch)
        
    def forward(self,word_seq):
        word_embeddings = self.src_word_emb(word_seq)
        word_pos = self.get_absolute_pos(word_seq)
        word_pos_encoded = word_embeddings + self.position_enc(word_pos)
        return word_pos_encoded
        
    
        

