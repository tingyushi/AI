from transformer import GPTLanguageModel, MAX_ITER
from utils import encoder, decoder
import torch
import json

if __name__ == '__main__':
    with open("transformer_model/chartoidx.json", "r") as fp:
        chartoidx = json.load(fp)
    idxtochar = {idx : char for char, idx in chartoidx.items()}

    # load mode
    model = GPTLanguageModel()
    model.load_state_dict(torch.load('transformer_model/transformer_model'))
    model = model.to('cuda')
    # ask user input 
    string = input("Please enter text: ")

    # encode string
    encoded_string = encoder(string, chartoidx)
    encoded_string = torch.tensor(encoded_string)
    encoded_string = torch.unsqueeze(encoded_string, 0)
    encoded_string = encoded_string.to('cuda')
    yhat = model.generate(encoded_string, MAX_ITER)

    # decode string
    decoded_string = torch.Tensor.tolist(yhat)[0]
    decoded_string = decoder(decoded_string, idxtochar)

    print()
    print("Generated Text:")
    print(decoded_string)
