# File to house all matplot related functions

from matplotlib import pyplot as plt

def showimg(state):
    plt.imshow(state)
    plt.show()

def show_framestack(state):
    plt.figure(figsize=(10,8))
    for i in range(state.shape[3]):
        plt.subplot(1,4, i+1)
        plt.imshow(state[0][:,:,i])
    plt.show()