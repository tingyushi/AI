import json
from utils import encoder, decoder
from bigram import BigramLanguageModel
import torch

if __name__ == '__main__':
    with open("bigram_model/chartoidx.json", "r") as fp:
        chartoidx = json.load(fp)
    vocab_size = len(chartoidx)
    idxtochar = {idx : char for char, idx in chartoidx.items()}

    # load mode
    bigram = BigramLanguageModel(vocab_size)
    bigram.load_state_dict(torch.load('bigram_model/bigram_model', 
                                      map_location=torch.device('cpu')))

    # ask user input 
    string = input("Please enter text: ")

    # encode string
    encoded_string = encoder(string, chartoidx)
    encoded_string = torch.tensor(encoded_string)
    encoded_string = torch.unsqueeze(encoded_string, 0)

    yhat = bigram.generate(encoded_string, 100)

    # decode string
    decoded_string = torch.Tensor.tolist(yhat)[0]
    decoded_string = decoder(decoded_string, idxtochar)

    print()
    print("Generated Text:")
    print(decoded_string)