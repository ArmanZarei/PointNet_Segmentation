import numpy as np
import matplotlib.pyplot as plt
import torch
import matplotlib.animation as animation


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

def test_accuracy(model, test_dataloader, device):
    model.eval()
    test_acc = .0
    with torch.no_grad():
        for input, labels in test_dataloader:
            input, labels = input.to(device).squeeze().float(), labels.to(device)
            outputs, _, _ = model(input)
            test_acc += (labels == outputs.argmax(1)).sum().item() / np.prod(labels.shape)
        test_acc /= len(test_dataloader)
    
    return test_acc


# --------------------------- Save GIF --------------------------- #
class PointCloudVisualizer:
    def __R_x(self, theta):
        return np.array([
            [1, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta), np.cos(theta)]
        ])
    
    def __R_y(self, theta):
        return np.array([
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)]
        ])
    
    def __R_z(self, theta):
        return np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
        ])
    
    def save_visualization(self, points, labels, file_path):
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        plt.grid(True)
          
        ims = []
        for rot in [self.__R_x, self.__R_y, self.__R_z]:
            for theta in np.linspace(0, 2*np.pi, 20):
                rp = rot(theta).dot(points.T).T
                ims.append([ax.scatter(rp[:, 0], rp[:, 1], rp[:, 2], c=labels)])
        
        ani = animation.ArtistAnimation(fig, ims, blit=True)
        ani.save(file_path, writer='pillow', fps=10)