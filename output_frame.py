# File to house all matplot related functions

from matplotlib import pyplot as plt

def showimg(state):
    plt.imshow(state, cmap='gray')
    plt.show()

def show_framestack(state):
    plt.figure(figsize=(10,8))
    print(state.shape)
    count = 1
    for i in range(0, state.shape[3], 2):
        plt.subplot(1,4, count)
        count +=1
        plt.imshow(state[0][:,:,i], cmap='gray')
    plt.show()