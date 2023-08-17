import torch
import torch.nn as nn
from progressbar import progressbar
import numpy as np

def get_patches_helper(data, patch_size):
    '''
    from 28 * 28 * 1 --> 49 * 16 
    '''
    height, width, channel = data.shape
    res = torch.zeros((int(height /  patch_size) ** 2, 
                       patch_size ** 2), 
                       dtype=torch.float)
    res_idx = 0
    for i in range(0, height, patch_size):
        for j in range(0, width, patch_size):
            temp = data[i:i+patch_size, j:j+patch_size, :]
            temp = temp.flatten()
            res[res_idx] = temp
            res_idx += 1
    return res


def get_patches(data, patch_size):
    '''
    data: shape is (number of pictures, 28, 28, 1)
    let the patch size be (4, 4) and now we have 7 patches
    return: shape (number of pictures, 7x7, 4x4)
    '''
    num_pic, height, width, channel = data.shape
    res = torch.zeros((num_pic, 
                       int(height /  patch_size) ** 2, 
                       patch_size ** 2), 
                       dtype=torch.float)

    for i in  range(num_pic):
        res[i] = get_patches_helper(data[i], patch_size)

    return res



# Linear Projection of Flattened Patches
class LinearProjection(nn.Module):
    def __init__(self, patch_dim, token_dim):
        super().__init__()
        self.linearprojection = nn.Linear(patch_dim, token_dim)
        self.class_token = nn.Parameter(torch.rand(1, token_dim))
        
    def forward(self, x):  
        lp = self.linearprojection(x)
        out = torch.cat([self.class_token, lp])
        return out
    

def get_positional_embedding(num_of_token, token_dim):
    pos_emb = torch.ones((num_of_token, token_dim), dtype=torch.float)
    for i in range(num_of_token):
        for j in range(token_dim):
            if j % 2 == 0:
                temp = torch.tensor(i / (10000**(j / token_dim)))
                pos_emb[i, j] = torch.sin(temp)
            else:
                temp = torch.tensor(i / (10000**( (j - 1) / token_dim)))
                pos_emb[i, j] = torch.cos(temp)
    return nn.Parameter(pos_emb)


class Head(nn.Module):
    def __init__(self, token_dim, head_size):
        super().__init__()
        self.token_dim = token_dim
        self.head_size = head_size
        self.key = nn.Linear(token_dim, head_size)
        self.value = nn.Linear(token_dim, head_size)
        self.query = nn.Linear(token_dim, head_size)
    
    def forward(self, x):
        k = self.key(x) ; v = self.value(x) ; q = self.query(x)
        temp = q @ k.T
        temp /= (self.head_size ** 0.5)
        weights = nn.functional.softmax(temp, dim=-1)
        out = weights @ v
        return out
    
class MultiHead(nn.Module):
    def __init__(self, token_dim, n_head):
        super().__init__()
        
        self.token_dim = token_dim
        self.n_head = n_head
        self.head_size = int(token_dim / n_head)
        self.heads = nn.ModuleList([Head(self.token_dim, self.head_size) for _ in range(self.n_head)])
    
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return out


class Block(nn.Module):
    def __init__(self, token_dim, n_head):
        super().__init__()
        self.token_dim = token_dim
        self.n_head = n_head
        self.head_size = int(token_dim / n_head)

        self.ln1 = nn.LayerNorm(token_dim)
        self.ln2 = nn.LayerNorm(token_dim)
        self.mha = MultiHead(token_dim, n_head)
        self.mlp = nn.Sequential(nn.Linear(token_dim, token_dim * 4),
                                 nn.GELU(),
                                 nn.Linear(token_dim * 4, token_dim))
    
    def forward(self, x):
        x = x + self.mha(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class Encoder(nn.Module):
    def __init__(self, token_dim, n_head, n_block):
        super().__init__()
        self.token_dim = token_dim ; self.n_head = n_head
        self.n_block = n_block ; self.head_size = int(token_dim / n_head)
        self.net = nn.Sequential()
        for _ in range(self.n_block):
            self.net.append(Block(self.token_dim, self.n_head))
        
    def forward(self, x):
        out = self.net(x)
        return out


class VIT(nn.Module):
    def __init__(self, token_dim, n_head, n_block, patch_size, height, width, channel, n_class, device):
        super().__init__()
        self.token_dim = token_dim ; self.n_head = n_head
        self.n_block = n_block ; self.patch_size = patch_size
        self.channel = channel ; self.height = height
        self.width = width ; self.n_of_token = int((self.height / self.patch_size) ** 2)
        self.n_class = n_class ; self.device = device

        self.linearprojection = LinearProjection(patch_dim= self.patch_size * self.patch_size * self.channel,
                                                 token_dim=self.token_dim)
        
        self.pos_embedding = get_positional_embedding(num_of_token=self.n_of_token + 1, 
                                                      token_dim=self.token_dim)

        self.encoder = Encoder(token_dim=self.token_dim,
                               n_head=self.n_head,
                               n_block=self.n_block)
        
        self.mlphead = nn.Linear(self.token_dim, self.n_class)

    def forward(self, x, y = None):
        '''
        shape of x: (height, width, channel)
        shape of y: (1,)
        '''

        # get x, y to correct shape and type
        x = torch.unsqueeze(x, 0)
        x = get_patches(x, self.patch_size)
        x = x[0]
        x = x.to(self.device)

        x = self.pos_embedding + self.linearprojection(x)
        x = self.encoder(x)

        out = self.mlphead(x[0])
        out = nn.functional.softmax(out, dim=-1)

        if y is None:
            loss = None
        else:
            y = y.item() 
            y = torch.tensor(y)
            y = y.to(self.device)
            loss = nn.functional.cross_entropy(out, y)

        return out, loss
    
    def predict(self, x):
        '''
        shape of x: (height, width, channel)
        '''
        out , _ = self(x)
        out = out.to('cpu')
        out = out.detach().numpy()
        return np.argmax(out)

@torch.no_grad()
def estimate_loss(model, x, y):
    model.eval()
    data_size = x.shape[0]
    losses = torch.zeros(data_size)
    correct = 0
    for i in progressbar( range(data_size) ):
        _, loss = model(x[i], y[i])
        losses[i] = loss

        if model.predict(x[i]) == y[i].item():
            correct += 1
        

    model.train()
    return losses.mean() , correct / data_size


def train(model, x_train, y_train, x_test, y_test, epoch, lr):
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    
    print()
    loss, accu = estimate_loss(model, x_train, y_train)
    print(f"Loss and Accuracy on training set before training: {loss}, {accu}")
    loss, accu = estimate_loss(model, x_test, y_test)
    print(f"Loss and Accuracy on testing  set before training: {loss}, {accu}")
    print()
    
    for _ in progressbar( range(epoch) ):
        for j in range( x_train.shape[0]) :
            x = x_train[j] ; y = y_train[j]
            _, loss = model(x, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    loss, accu = estimate_loss(model, x_train, y_train)
    print(f"Loss and Accuracy on training set after training: {loss}, {accu}")
    loss, accu = estimate_loss(model, x_test, y_test)
    print(f"Loss and Accuracy on testing  set after training: {loss}, {accu}")
    print()

    print("Saving model ...")
    torch.save(model.state_dict(), 'trained_model/model')
    print("Model Saved ...")


def parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
