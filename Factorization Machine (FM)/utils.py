import matplotlib.pyplot as plt
from typing import List


def plot_loss_graph(train_loss: List, test_loss: List):
    plt.plot(train_loss, label='train loss', color='#1ABDE9')
    plt.plot(test_loss, label='test loss', color='#F36E8E')
    plt.title('Factorization Machines Loss graph')
    plt.xlabel('Epoch')
    plt.ylabel('BCE Loss')
    plt.legend()
    plt.show()
