import matplotlib.pyplot as plt

def show_image(data, label):
    plt.imshow(data, 'gray')
    plt.title(str(label))
    plt.savefig('image.png')