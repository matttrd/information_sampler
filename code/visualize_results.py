import numpy as np
import imageio
import pickle as pkl 
import matplotlib.pyplot as plt

with open('./histograms_our_sampler.pickle', 'rb') as handle:
    hist = pkl.load(handle)

bin_edges = hist['bin_edges']
num_classes = 10
y_max = 1

def plot_for_offset(current_histogram, bin_edges, label, epoch, y_max):
    total_count = np.sum(current_histogram)
    current_histogram = current_histogram / total_count
    fig, ax = plt.subplots(figsize=(10,5))
    ax.bar(bin_edges[:-1], current_histogram, width = 0.01)
    ax.set(xlabel='Weight', ylabel='Perc', title='Label {}, Epoch {}'.format(label, epoch))
    ax.grid()
    # Used to keep the limits constant
    ax.set_ylim(0, y_max)
    ax.set_xlim(0, 1)
    # Used to return the plot as an image rray
    fig.canvas.draw()       # draw the canvas, cache the renderer
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image  = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    return image

kwargs_write = {'fps':1.0, 'quantizer':'nq'}

for label in range(num_classes): 
    current_label_histograms = hist[str(label)]
    imageio.mimsave('./hist_label_' + str(label) + '.gif', [plot_for_offset(current_label_histograms[i], bin_edges, label, i+1, y_max) 
                                                            for i in range(len(current_label_histograms))], fps=2)
    





