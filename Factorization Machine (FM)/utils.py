import matplotlib.pyplot as plt
from typing import List


def plot_loss_graph(train_loss: List, test_loss: List):
    fig, train_ax = plt.subplots()
    train_ax.set_xlabel('Epoch')
    train_ax.set_ylabel('BCE Loss', color='#1ABDE9')
    train_ax.plot(train_loss, label='train loss', color='#1ABDE9')
    train_ax.tick_params(axis='y', labelcolor='#1ABDE9')

    test_ax = train_ax.twinx()
    test_ax.set_ylabel('AUCROC', color='#F36E8E')
    test_ax.plot(test_loss, label='test loss', color='#F36E8E')
    test_ax.tick_params(axis='y', labelcolor='#F36E8E')

    plt.legend()
    plt.show()
