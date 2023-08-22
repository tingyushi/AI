from hparams import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def read_csv(file_path):
    df = pd.read_csv(file_path)
    num_of_pics = len(df.axes[0])
    num_of_clos = len(df.axes[1])

    x = np.ones((num_of_pics, HEIGHT, WIDTH), dtype=np.uint8)
    y = np.ones(num_of_pics, dtype=np.uint8)

    for i in range(num_of_pics):
        y[i] = df.iloc[i, 0]
        temp = np.array(df.iloc[i, 1 : num_of_clos])
        temp = temp.reshape(HEIGHT, WIDTH)
        x[i] = temp

    return x, y
