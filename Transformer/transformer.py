import torch
import torch.nn as nn
import json
import collections
collections.Iterable = collections.abc.Iterable
from progressbar import progressbar
from utils import *


TEXT_PATH = 'text.txt'
SAVED_MODEL_PATH = 'transformer_model/transformer_model'
BATCH_SIZE = 64 
BLOCK_SIZE = 256 # how many characters in a batch
ITER_NUM = 2500 # number of gradient descent
LEARNING_RATE = 3e-4
D_MODEL = 384 # embedding dimension
N_HEAD = 6 # number of heads in multi-head attention
BLOCK_NUM = 6 # Number of decoding blocks
DROPOUT_PROB = 0.2
HEAD_SIZE = int(D_MODEL / N_HEAD)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EPOCH = 50
MAX_ITER = 200 # used for generating text

# load data
text , chartoidx , idxtochar = load_data(TEXT_PATH)
VOCAB_SIZE = len(chartoidx)
with open("transformer_model/chartoidx.json", "w") as fp:
    json.dump(chartoidx, fp)



# get a random batch of data
def get_one_batch(data):
    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([data[i:i+BLOCK_SIZE] for i in ix])
    y = torch.stack([data[i+1:i+BLOCK_SIZE+1] for i in ix])
    return x, y

# single head attention
class SingleHead(nn.Module):
    def __init__(self):
        '''
        HEAD_SIZE : dimension of key, query, value
        '''
        super().__init__()
        self.key = nn.Linear(D_MODEL, HEAD_SIZE, bias=False)
        self.query = nn.Linear(D_MODEL, HEAD_SIZE, bias=False)
        self.value = nn.Linear(D_MODEL, HEAD_SIZE, bias=False)

        # lower triangular
        self.register_buffer('tril', torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))

        self.dropout = nn.Dropout(DROPOUT_PROB)

    def forward(self, x):
        '''
        shape of x: BATCH_SIZE, BLOCK_SIZE, D_MODEL
        '''

        B,T,C = x.shape
        
        # shape (BATCH_SIZE, BLOCK_SIZE, HEAD_SIZE)
        k = self.key(x)

        # shape (BATCH_SIZE, BLOCK_SIZE, HEAD_SIZE)
        q = self.query(x)

        # (BATCH_SIZE, BLOCK_SIZE, HEAD_SIZE) * (BATCH_SIZE, HEAD_SIZE, BLOCK_SIZE)
        # result: (BATCH_SIZE, BLOCK_SIZE, BLOCK_SIZE)
        wei = q @ k.transpose(-2,-1) / (k.shape[-1] ** 0.5)

        # (BATCH_SIZE, BLOCK_SIZE, BLOCK_SIZE)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))

        # (BATCH_SIZE, BLOCK_SIZE, BLOCK_SIZE)
        wei = nn.functional.softmax(wei, dim=-1) 
        wei = self.dropout(wei)

        # (BATCH_SIZE, BLOCK_SIZE, HEAD_SIZE)
        v = self.value(x)

        # (BATCH_SIZE, BLOCK_SIZE, BLOCK_SIZE) @ (BATCH_SIZE, BLOCK_SIZE, HEAD_SIZE) 
        # (BATCH_SIZE, BLOCK_SIZE, HEAD_SIZE)
        out = wei @ v 
        return out



class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = nn.Sequential()
        for _ in range(N_HEAD):
            self.heads.append(SingleHead())
        self.proj = nn.Linear(HEAD_SIZE * N_HEAD, D_MODEL)
        self.dropout = nn.Dropout(DROPOUT_PROB)

    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out



class FeedFoward(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(D_MODEL, 4 * D_MODEL),
            nn.ReLU(),
            nn.Linear(4 * D_MODEL, D_MODEL),
            nn.Dropout(DROPOUT_PROB),
        )

    def forward(self, x):
        return self.net(x)



# combine multihead attention with feedforward
class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.multiheads = MultiHeadAttention()
        self.feedforward = FeedFoward()
        self.ln1 = nn.LayerNorm(D_MODEL)
        self.ln2 = nn.LayerNorm(D_MODEL)
    
    def forward(self, x):
        x = x + self.multiheads(self.ln1(x))
        x = x + self.feedforward(self.ln2(x))
        return x


class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(VOCAB_SIZE, D_MODEL)
        self.position_embedding_table = nn.Embedding(BLOCK_SIZE, D_MODEL)
        self.blocks = nn.Sequential()
        for _ in range(BLOCK_NUM):
            self.blocks.append(Block())
        self.ln = nn.LayerNorm(D_MODEL)
        self.lm_head = nn.Linear(D_MODEL, VOCAB_SIZE)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        '''
        idx: shape(BATCH_SIZE, BLOCK_SIZE)
        '''
        B, T = idx.shape
        
        tok_emb = self.token_embedding_table(idx) # shape (BATCH_SIZE, BLOCK_SIZE, D_MODEL)
        pos_emb = self.position_embedding_table(torch.arange(T, device= DEVICE)) # shape (BLOCK_SIZE, D_MODEL)
        x = tok_emb + pos_emb # (BATCH_SIZE, BLOCK_SIZE, D_MODEL)
        x = self.blocks(x) # (BATCH_SIZE, BLOCK_SIZE, D_MODEL)
        x = self.ln(x) # (BATCH_SIZE, BLOCK_SIZE, D_MODEL)
        logits = self.lm_head(x) # (BATCH_SIZE, BLOCK_SIZE, VOCAB_SIZE)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = nn.functional.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):

        for _ in range(max_new_tokens):
            idx_cond = idx[:, 0:BLOCK_SIZE]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] 
            probs = nn.functional.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1) 
            idx = torch.cat((idx, idx_next), dim=1) 
        return idx


def parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



@torch.no_grad()
def estimate_loss(model, data):
    model.eval()
    losses = torch.zeros(MAX_ITER, dtype=torch.float)

    for idx in range(MAX_ITER):
        x , y = get_one_batch(data)
        _, loss = model(x, y)
        losses[idx] = loss.item()
    
    model.train()

    return losses.mean()


if __name__ == '__main__':
    
    # split text
    train_text = text[0:int(0.8 * len(text))]
    test_text = text[int(0.8 * len(text)) :]

    # encode 
    train_data = encoder(train_text, chartoidx) ; train_data = torch.tensor(train_data, dtype=torch.int64)
    test_data = encoder(test_text, chartoidx)   ; test_data = torch.tensor(test_data, dtype=torch.int64)

    gpt = GPTLanguageModel()
    print()
    print(f"Number of parameters: {parameters(gpt)}")
    print()
    
    # move to device
    print(f"Training on device: {DEVICE}")
    print()
    train_data = train_data.to(DEVICE) 
    test_data = test_data.to(DEVICE)
    gpt = gpt.to(DEVICE)
    
    optimizer = torch.optim.AdamW(gpt.parameters(), lr=LEARNING_RATE)

    print(f"Loss on training set before training: {estimate_loss(gpt, train_data)}")
    print(f"Loss on testing  set before training: {estimate_loss(gpt, test_data)}")
    print()
    

    for i in progressbar( range(ITER_NUM) ):
        x, y = get_one_batch(train_data)
        _, loss = gpt(x, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # used for adjusting hyperparameters
        '''
        if (i + 1) % 100 == 0:
            print()
            print(f"Epoch: {i + 1}")
            print(f"Loss on training set: {estimate_loss(gpt, train_data)}")
            print(f"Loss on testing  set: {estimate_loss(gpt, test_data)}")
            print()
        '''

    print(f"Loss on training set after training: {estimate_loss(gpt, train_data)}")
    print(f"Loss on testing  set after training: {estimate_loss(gpt, test_data)}")
    print()
    
    # save model
    torch.save(gpt.state_dict(), SAVED_MODEL_PATH)
