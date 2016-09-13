import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import Tkinter as Tk
import networkx as nx
import copy
import numpy as np
from genome import Genome

#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.

# If it is not possible or desirable to put the notice in a particular
# file, then You may include the notice in a location (such as a LICENSE
# file in a relevant directory) where a recipient would be likely to look
# for such a notice.

# You may add additional accurate notices of copyright ownership.

def unpart1by1(n):
    n &= 0x55555555
    n = (n ^ (n >> 1)) & 0x33333333
    n = (n ^ (n >> 2)) & 0x0f0f0f0f
    n = (n ^ (n >> 4)) & 0x00ff00ff
    n = (n ^ (n >> 8)) & 0x0000ffff
    return n


def deinterleave2(n):
    return unpart1by1(n), unpart1by1(n >> 1)


# Creates a two-dimensional
# #Z-order curve by using the de-interleaving method
def z_layout(n):
    pos = {}
    for i in range(0, n):
        pos[i] = deinterleave2(i)
    return pos

# Parameters for drawing graphs
node_size=100
node_color='blue'
node_alpha=0.3
node_text_size=12
edge_color='blue'
edge_alpha=0.3
edge_thickness=1
edge_text_pos=0.8
text_font='sans-serif'

# the population size
POP_SIZE = 9

#the weight threshold
WEIGHT_THRESHOLD = 0.1

frames = [None]*POP_SIZE
figures = [None]*POP_SIZE
axes = [None]*POP_SIZE
G = [None]*POP_SIZE
pos = [None]*POP_SIZE
xlim = [None]*POP_SIZE
ylim = [None]*POP_SIZE
buttons = [None]*POP_SIZE

genomes = [None]*POP_SIZE


def mutate_graph(i):
    for j in range(POP_SIZE):
        if j != i:
            genomes[j] = copy.deepcopy(genomes[i])
            genomes[j].mutate()
            plt.close(figures[j])
            next_graph(j)


def next_graph(i):

    frames[i] = Tk.Frame(root)
    frames[i].grid(row=i / 3, column=i % 3)
    figures[i] = plt.figure(figsize=(5, 4))

    axes[i] = figures[i].add_subplot(111)
    plt.axis('off')

    phenotype = genomes[i].get_phenotype()
    phenotype[phenotype < WEIGHT_THRESHOLD] = 0
    G[i] = nx.from_numpy_matrix(phenotype, create_using=nx.DiGraph())

    pos[i] = z_layout(np.ma.size(phenotype, 0))

    colorList = []
    for j in G[i].edges():
        a, b = j
        colorVal = G[i].edge[a][b]['weight']
        colorList.append(colorVal)

    # nx.draw_networkx(G[i], pos=pos[i], ax=axes[i])
    nx.draw_networkx_nodes(G[i], pos[i], node_size=node_size,
                           alpha=node_alpha, node_color=node_color)
    nx.draw_networkx_edges(G[i], pos[i], width=colorList, edge_color=colorList, edge_cmap=plt.cm.RdYlGn, arrows=True)
    nx.draw_networkx_labels(G[i], pos[i], font_size=node_text_size,
                            font_family=text_font)

    xlim[i] = axes[i].get_xlim()
    ylim[i] = axes[i].get_ylim()

    canvas = FigureCanvasTkAgg(figures[i], master=frames[i])
    canvas.show()
    canvas.get_tk_widget().pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)

    buttons[i] = Tk.Button(frames[i], text="choose as parent", command=lambda i=i: mutate_graph(i))
    buttons[i].pack()

# The first genome is initialized to default
genomes[0] = Genome()

# initialize the genomes
for i in range(1, POP_SIZE):
    genomes[i] = copy.deepcopy(genomes[0])
    genomes[i].mutate()

root = Tk.Tk()
root.wm_title("CAMPN Breeder")
root.wm_protocol('WM_DELETE_WINDOW', root.quit())

# initialize and draw the initial window
for i in range(POP_SIZE):
    next_graph(i)


Tk.mainloop()
