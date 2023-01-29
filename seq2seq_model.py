# following tutorial from https://youtu.be/EoGUlvhRYpk

import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.datasets import Multi30k # Multi30k = popular german to english words dataset

# these imports are from a separate video he made...
# from utils import translate_sentence, bleu, save_checkpoint, load__checkpoint
'''
# TODO: I am having an IMPORT ERROR

# to solve it I need to download an old version of torchtext, 
# but I'm not sure how to do that so I'm going to try to find another way

from torchtext.legacy.data import Field, BucketIterator # for pre processing
'''
# from torchtext.legacy.data import Field, BucketIterator # for pre processing
# from torchtext.data.legacy import Field, BucketIterator
# from torchtext.data.field import Field, BucketIterator
import numpy as np
import spacy # tokenizer
import random
from torch.utils.tensorboard import SummaryWriter # to print to tensorboard


#######################
# FORMATTING ONE SMALL THING
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')




#######################################
# DATA PRE PROCESSING

spacy_ger = spacy.load('de') # loading the data
spacy_eng = spacy.load('en')

# creating tokenizers, which split sentences into words
# e.g. 'Hello my name is' -> ['Hello', 'my', 'name', 'is']
def tokenizer_ger(text):
    return [tok.text for tok in spacy_ger.tokenizer(text)]

def tokenizer_eng(text):
    return [tok.text for tok in spacy_eng.tokenizer(text)]


"""
Some stuff w/ Field, setting up the training/test data, and building the english & german vocabs here 
^^ check mins 3-6

"""


class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, p): # p stands for drop out
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

         # we're gonna run the embedding on the input, then we run the rnn on the embedding
        self.dropout = nn.Dropout(p)
        self.embedding = nn.Embedding(input_size, embedding_size)
        # LSTM notes
        # input = embedding_size
        # output = hidden_size
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=p)

        def forward(self, x): # explanation in tutorial at 9 mins
            # x shape: (seq_length, N), where N = batch_size

            embedding = self.dropout(self.embedding(x))
            # embedding shape: (seq_length, N, embedding_size) -- added another dimension to tensor to define the embedding size


            # self.rnn(embedding) -- THIS LINE WAS IN HIS TUTORIAL BUT DISAPPEARED WITHOUT EXPLANATION around 10:30
            outputs, (hidden, cell) = self.rnn(embedding)

            return hidden, cell


class Decoder(nn.Module):

    # input size here will be size of english vocab
    # output size = input_size, since for a 10k word english vocabulary, any word in english is mapping to another word in german where the vocabulary is the same size
    def __init__(self, input_size, embedding_size, hidden_size, 
                output_size, num_layers, p): # p stands for drop out
        
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.dropout = nn.Dropout(p)
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=p)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden, cell):
        # shape of x: (N), but we want (1, N)
        # decoder will predict 1 word at a time, so given the previous,hidde, & cell state & the previous predicted word
        # predicts word by word -- N batches of a single word at the same time
        # in contrast to the encoder, which took entire sentences at a time ~ notice that it says (seq_length, N) in the encoder's fwd pass
        # encoder takes in sentences, decoder spits out word by word

        x = x.unsqueeze(0) # added 1 dimension

        embedding = self.dropout(self.embedding(x))
        # embedding shape: (1, N, embedding_size)
        
        # now: sending it into the LSTM
        outputs, (hidden, cell) = self.rnn(embedding, (hidden, cell))
        # shape of outputs: (1, N, hisdden_size)

        predictions = self.fc(outputs)
        # shape of predictions: (1, N, length_of_vocab)
        # predictions will be sent to loss function later on

        # now we want to remove the 1 dimension so we do:
        predictions = predictions.squeeze(0)

        return predictions, hidden, cell

###########################################################
# BUILDING THE MODEL -- putting it all together
class Seq2Seq(nn.Module): # combining encoder & decoder
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    # sending in the source sentence & the target sentence (aka training data & label)
    def forward(self, source, target, teacher_force_ratio=0.5):
        # note that teacher_force_ratio =0.5 means that 1/2 of the time the model will use the prediction & half of the time it will use the data label to taech the model better
        batch_size = source.shape[1]
        # (trg_len, N)
        target_len = target.shape[0]
        target_vocab_size = len(english.vocab) # NOTE: come back to this later, I left it out bc it has to do w/ field import

        outputs = torch.zeros(target_len, batch_size, target_vocab_size).to(device)

        # encoder takes in the tokenized English vocab hidden & cell 
        # decoder takes in hidden & cell 
        # so we define hidden & cell
        hidden, cell = self.encoder(source)

        # Grab start token ??
        x = target[0]

        for t in range(1, target_len):
            output, hidden, cell = self.decoder(x, hidden, cell).to(device) # added at 25:55

            outputs[t] = output

            # output: (N, english_vocab_size)
            best_guess = output.argmax(1) # ??

            # next input will be target word if random # < best guess
            x = target[t] if random.random() < teacher_force_ratio else best_guess

        return outputs


###########################################################
'''
At this point, we have:
- pre processed data
- built the encoder
- built the decoder
- built the model

Now we will:
- train the model
- test the model
- save the model 
'''

###########################################################
# TRAINING THE MODEL

# training hyper-parameters
num_epochs = 20
learning_rate = 0.001
batch_size = 64

# Model hyper-parameters
load_model = Falsedevice = torch.device('cuda' if torch.cuda.is_available else 'cpu')
input_size_encoder = len(german.vocab)
input_size_decoder = len(english.vocab)
output_size = len(english.vocab)
encoder_embedding_size = 150 # 100-300 is a good # for the encoder & decoder embedding sizes
decoder_embedding_size = 150
hidden_size = 1024 # look up research papers to see what model hyper parameters are common
num_layers = 2
enc_dropout = 0.5
dec_dropout = 0.5

# code for Tensorboard 
writer = SummaryWriter(f'runs/loss_plot')
step = 0

# THIS USES BUCKET ITERATOR - USE SOMETHING ELSE HERE
# TODO: ^^^^^
train_iterator, valid_iterator, test_iterator = BucketIterator.splits((
    train_data, validation_data, test_data),
    batch_size=batch_size,
    sort_within_batch = True, # this batches sentences of similar length together, this line & the one below it is important
    sort_key = lambda x: len(x.src),
    device=device)

encoder_net = Encoder(input_size_encoder, encoder_embedding_size, hidden_size, num_layers, enc_dropout).to(device)

decoder_net = Decoder(input_size_decoder, decoder_embedding_size, hidden_size, output_size, num_layers, dec_dropout).to(device)


model = Seq2Seq(encoder_net, decoder_net).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

pad_idx = english.vocab.stoi['<pad>']
 # stoi = 'stream to index'
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)


###########################################################
# LOADING THE MODEL

if load_model:
    load_checkpoint(torch.load('my_checkpoint.pth.ptar'), model, optimizer)

# testing how translation of a single sentence improves with each epoch!
sentence = 'ein boot mit mehreren männern darauf wird von einem groβben pferdegespann an ufer gezogen'


for epoch in range(num_epochs):
    print(f'Epoch [{epoch} / {num_epochs}]')

    checkpoint = {'state_dict':model.state_dict(), 'optimizer':optimizer.state_dict()}
    save_checkpoint(checkpoint)

    model.eval()

    translated_sentence = translated_sentence(model, sentence, german, english, device, max_length=50)

    print(f'Untranslated example sentence \n{sentence}')
    print(f'Translated example sentence \n{translated_sentence}')

    model.train()

    for batch_idx, batch in enumerate(train_iterator):
        input_data = batch.src.to(device)
        target = batch.trg.to(device)

        output = model(input_data, target)
        # output shape: (trg_len, batch_size, output_dim)

        output = output[1:].reshape(-1, output.shape[2])
        target = target[1:].reshape(-1)

# not sure which is right
        # torch.optim.zero_grad
        optimizer.zero_grad() # the video had this one
    
        loss = criterion(output, target)

        loss.backward()

        # to make sure the gradients don't explode
        torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=1)
        optimizer.step()

        writer.add_scalar('Training Loss', loss, global_step=step)
        step += 1

