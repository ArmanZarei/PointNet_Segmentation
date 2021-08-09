import numpy as np
import matplotlib.pyplot as plt


def read_pts(file):
    return np.genfromtxt(file)

def read_seg(file):
    return np.genfromtxt(file, dtype=(int))

def training_process_plot_save(train_loss_arr, val_loss_arr, train_accuracy_arr, val_accuracy_arr, save_dir='images/training.png'):
    plt.figure(figsize=(20, 8))
    plt.subplot(1, 2, 1).set_title("Loss / Epoch")
    plt.plot(train_loss_arr, label='Train')
    plt.plot(val_loss_arr, label='Validation')
    plt.legend()
    plt.subplot(1, 2, 2).set_title("Accuracy / Epoch")
    plt.plot(train_accuracy_arr, label='Train')
    plt.plot(val_accuracy_arr, label='Validation')
    plt.legend()
    plt.savefig(save_dir)