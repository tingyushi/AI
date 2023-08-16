import torch
import torch.nn as nn
import json
import collections
collections.Iterable = collections.abc.Iterable
from progressbar import progressbar
from utils import *

class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, x, y = None):
        '''
        x: shape (batch_size, block_size)
        y: shape (batch_size, block_size)
        '''

        # the shape of yhat is (batch_size, block_size, embedding_dimension)
        yhat = self.token_embedding_table(x)

        if y is None: # used for inference
            loss = None
        else: # used for training
            B, T, C = yhat.shape
            yhat = yhat.view(B*T, C)
            y = y.view(B*T)
            loss = nn.functional.cross_entropy(yhat, y)

        return yhat, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions shape(batch_size, block_size, embedding_dimension)
            logits, _ = self(idx)
            
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (batch_size, embedding_dimension)
            
            # apply softmax to get probabilities 
            probs = nn.functional.softmax(logits, dim=-1) # shape (batch_size, embedding_dimension)
            
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (batch_size, embedding_dimension)
            
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


@torch.no_grad()
def estimate_loss(model, x, y):
    model.eval()

    num_of_batch = x.shape[0]
    losses = torch.zeros(num_of_batch, dtype=torch.float)

    for batch_idx in range(num_of_batch):
        _, loss = model(x[batch_idx], y[batch_idx])
        losses[batch_idx] = loss.item()
    
    model.train()

    return losses.mean()

if __name__ == "__main__":
    TEXT_PATH = 'text.txt'
    SAVED_MODEL_PATH = 'bigram_model/bigram_model'
    BATCH_SIZE = 4 
    BLOCK_SIZE = 8 # how many characters in a batch
    EPOCH = 5
    LEARNING_RATE = 1e-2

    # load data
    text , chartoidx , idxtochar = load_data(TEXT_PATH)
    vocab_size = len(chartoidx)
    with open("bigram_model/chartoidx.json", "w") as fp:
        json.dump(chartoidx, fp)
    with open("bigram_model/idxtochar.json", "w") as fp:
        json.dump(idxtochar, fp)

    # split text
    train_text = text[0:int(0.8 * len(text))]
    test_text = text[int(0.8 * len(text)) :]

    # encode 
    train_data = encoder(train_text, chartoidx)
    test_data = encoder(test_text, chartoidx)

    # get batches
    x_train , y_train = get_batch(train_data, BATCH_SIZE, BLOCK_SIZE)
    x_test, y_test = get_batch(test_data, BATCH_SIZE, BLOCK_SIZE)

    print(f"number of batches = {x_train.shape[0]}")

    # create model
    bigram = BigramLanguageModel(vocab_size)
    
    # move data and model to mps
    # mps is slower than cpu , don't know why
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print()
    print("Training on device: " + device)
    print()
    x_train = x_train.to(device) 
    y_train = y_train.to(device)
    x_test = x_test.to(device) 
    y_test = y_test.to(device)
    bigram.to(device)
    
    
    # check out model parameters
    print("Model Parameters")
    for param_tensor in bigram.state_dict():
        print(param_tensor, "\t", bigram.state_dict()[param_tensor].size())
    print()

    # train the model
    optimizer = torch.optim.AdamW(bigram.parameters(), lr=LEARNING_RATE)

    print(f"Loss on training set before training: {estimate_loss(bigram, x_train, y_train)}")
    print(f"Loss on testing  set before training: {estimate_loss(bigram, x_test, y_test)}")
    print()

    for epoch in range(EPOCH):
        print(f"Epoch {epoch + 1}")
        for i in progressbar( range(x_train.shape[0]) ):
            x = x_train[i] ; y = y_train[i]
            #print(f"shape of x: {x.shape}")
            #print(f"shape of y: {y.shape}")
            _ , loss = bigram(x, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        print(f"Loss on training set: {estimate_loss(bigram, x_train, y_train)}")
        print(f"Loss on testing  set: {estimate_loss(bigram, x_test, y_test)}")
        print()
    
    # save model
    torch.save(bigram.state_dict(), SAVED_MODEL_PATH)

