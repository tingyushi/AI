import torch

def load_data(path):
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()
    # text has type str
    
    # get unique chars
    chars = list(set(text))
    chars = sorted(chars)
    # get chartoidx and idxtochar
    chartoidx = {} ; idxtochar = {}
    for idx, char in enumerate(chars):
        chartoidx[char] = idx
        idxtochar[idx] = char
    return text , chartoidx , idxtochar



# given a string, return a list of numbers
def encoder(str, chartoidx):
    res = []
    for c in str:
        res.append(chartoidx[c])
    return res



# given a list of number, return a string
def decoder(indices, idxtochar):
    res = ''
    for idx in indices:
        res += idxtochar[idx]
    return res



# convert data to batches
def get_batch(data, batch_size, block_size):
    num_of_batch = int((len(data) - 1) / (block_size * batch_size))
    x = torch.zeros(num_of_batch, batch_size, block_size, dtype=torch.int64)
    y = torch.zeros(num_of_batch, batch_size, block_size, dtype=torch.int64)

    start_index = 0
    for i in range(num_of_batch):
        for j in range(batch_size):
            x[i, j, :] = torch.tensor(data[start_index : start_index + block_size])
            y[i, j, :] = torch.tensor(data[start_index + 1 : start_index + block_size + 1])
            start_index += 1
    
    return x, y
