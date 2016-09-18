import numpy as np
import networkx as nx
import random
import matplotlib.pyplot as plt
import scipy.signal
import pygraphviz
import pydot

#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.

# If it is not possible or desirable to put the notice in a particular
# file, then You may include the notice in a location (such as a LICENSE
# file in a relevant directory) where a recipient would be likely to look
# for such a notice.

# You may add additional accurate notices of copyright ownership.

# Written by Stefano Palmieri in August-September 2016


# This class stores the genome as a weighted directed graph representing
# a Compositional Adjacency Matrix Producing Network. The weighted directed
#  graph is produced from Nodes and Links. Nodes can output a constant
# matrix (tensor) or produce the matrix product, transpose, convolution, or
# Kronecker product of the input matrices. Links are floating point values
# that are use for scalar multiplication of matrices outputted from Nodes.
class Genome(object):

    # Probability of mutating a new Node
    P_NEW_NODE = 0.15

    # Probability of mutating a new Link
    P_NEW_LINK = 0.15

    # Probability of mutating existing node
    P_NODE = 0.3

    # Probability of mutating existing link
    P_LINK = 0.3

    # Weights of a new added link
    N_LINK_WEIGHT = 1.0

    # Dictionary for types of Nodes
    types = {
        0: 'constant',
        1: 'kronecker',
        2: 'elem_mult'
        }

    # Sizes for matrices in the constant node type
    CONSTANT_ROW_SIZE = 2
    CONSTANT_COL_SIZE = 2

    # Mean and standard deviation for weight mutations
    MUTATE_MEAN = 0.1
    MUTATE_STD = 0.1

    def __init__(self):
        self.network = nx.DiGraph()
        # node with 0 id is the output
        self.network.add_node(0, type='output',
                              constant=np.matrix([[0.0, 0.0], [0.0, 0.0]], np.float16))
        #self.network.add_node(1, type='kronecker')
        #self.network.add_edge(1, 0, weight=1.0, left=True)
        self.network.add_node(1, type='constant',
                              constant=np.matrix([[1.0, 0.0], [0.0, 1.0]], np.float16))
        self.network.add_edge(1, 0, weight=1.0, left=True)


    # Mutations can add a Node, add a Link, change the weight of a Link,
    # change the function of a Node.
    def mutate(self):
        # If mutation occurs, mutate a new node
        rand = random.random()
        if rand <= self.P_NEW_NODE:
            # print "new node"
            self.mutate_new_node()
        # If mutation occurs, mutate a new link
        rand = random.random()
        if rand <= self.P_NEW_LINK:
            # print "new link"
            self.mutate_new_link()
        # If mutation occurs, mutate an existing node
        rand = random.random()
        if rand <= self.P_NODE:
            # print "node"
            self.mutate_node()
        # If mutation occurs, mutate an existing link
        rand = random.random()
        if rand <= self.P_LINK:
            # print "link"
            self.mutate_link()

    # Randomly mutates one of the existing nodes in the CAMPN.
    def mutate_node(self):

        # keep picking a node until a constant node is found
        while True:
            # pick a random node that's not the output node
            node = random.randrange(1, self.network.number_of_nodes())
            if self.network.node[node]['type'] == 'constant':
                break

        # pick a type of function to mutate to
        # func = self.types[random.randrange(0, len(self.types))]

        # self.network.node[node]['type'] = func

        # mutate one of it's constant's elements
        constant = self.network.node[node]['constant']
        rand_row = random.randrange(0, self.CONSTANT_ROW_SIZE)
        rand_col = random.randrange(0, self.CONSTANT_COL_SIZE)

        # Mutate by Gaussian random variable
        change = np.random.normal(self.MUTATE_MEAN, self.MUTATE_STD)

        # apply the change
        constant[rand_row, rand_col] += change
        self.network.node[node]['constant'] = constant

    # Randomly mutate an existing link by modifying the weight
    def mutate_link(self):
        # Add a Gaussian random variable to existing weight
        weight_change = np.random.normal(self.MUTATE_MEAN, self.MUTATE_STD)

        # Select a random edge
        rand = random.randrange(0, len(self.network.edges()))
        start, end = self.network.edges()[rand]
        self.network[start][end]['weight'] += weight_change

    # Create a new random Node and add it to the Genome.
    def mutate_new_node(self):
        # Randomly select the function for this node
        # func = self.types[random.randrange(0, len(self.types))]

        # Randomly select a node from the existing network
        end = random.randrange(0, self.network.number_of_nodes())

        if not self.network.node[end]['type'] == 'constant':
            # if the end node is kronecker/output, add a constant as a predecessor
            matrix = np.random.rand(self.CONSTANT_ROW_SIZE, self.CONSTANT_COL_SIZE).astype(np.float16)
            self.network.add_node(self.network.number_of_nodes(),
                                  type='constant', constant=matrix)
            self.network.add_edge(self.network.number_of_nodes()-1,
                                  end, weight=self.N_LINK_WEIGHT,
                                  left=bool(random.getrandbits(1)))

        elif self.network.node[end]['type'] == 'constant':
            # if the node is a constant, place a kronecker in between
            matrix = self.network.node[end]['constant']

            if random.random() < 0.5:
                self.network.node[end]['type'] = 'kronecker'
            else:
                self.network.node[end]['type'] = 'elem_mult'
            self.network.node[end]['constant'] = None
            self.network.add_node(self.network.number_of_nodes(),
                                  type='constant', constant=matrix)
            self.network.add_edge(self.network.number_of_nodes()-1,
                                  end, weight=self.N_LINK_WEIGHT,
                                  left=bool(random.getrandbits(1)))

    # Create a new Link and add it to the Genome.
    def mutate_new_link(self):
        # select a random node to start a link from, don't include output node
        start = random.randrange(1, self.network.number_of_nodes())

        # Keep picking a terminal node until one that is not the
        # start link and not a constant type is found
        while True:
            end = random.randrange(0, self.network.number_of_nodes())
            if not self.network.node[end]['type'] == 'constant':
                break

        # First check this link doesn't already exist so we don't add it again.
        # Otherwise we can consider this mutations as failed if it does exist
        # (no mutation occurred).
        if not (start, end) in self.network.edges():
            # If the end node has no incoming links, make link primary.
            self.network.add_edge(start, end, weight=self.N_LINK_WEIGHT,
                                  left=bool(random.getrandbits(1)))

            # attempt to add the link. Do this checking that
            # the number of simple cycles does not become one or greater.
            # Otherwise we have to remove the link to keep the CTPPN acyclic
            # and can consider this a mutation that failed (no mutation occurred).
            cycles = len(list(nx.simple_cycles(self.network)))

            if cycles >= 1 or start == end:
                self.network.remove_edge(start, end)

    # recursively compute output for a node
    def get_phenotype(self, node=None):

        if node is None:
            return self.get_phenotype(0)

        else:
            # optimization for constant node so don't have to calculate sub
            # phenotype
            if self.network.node[node]['type'] == 'constant':
                return self.network.node[node]['constant']

            predecessors = list(self.network.predecessors(node))

            left_nodes = self.left_nodes(node, predecessors)

            left_matrix = np.zeros((self.CONSTANT_ROW_SIZE, self.CONSTANT_COL_SIZE))
            logit = np.zeros((self.CONSTANT_ROW_SIZE, self.CONSTANT_COL_SIZE))


            # remove the left nodes from the predecessor list
            for left in left_nodes:
                predecessors.remove(left)

            # left will be weighted sum of left node outputs
            for left in left_nodes:
                weight = self.network[left][node]['weight']
                left_output = weight * self.get_phenotype(left)

                # rescale smaller matrix if need be.
                while np.ma.size(left_output, 0) != np.ma.size(left_matrix, 0):
                    if np.ma.size(left_matrix, 0) < np.ma.size(left_output, 0):
                        left_matrix = np.kron(left_matrix,
                                        np.ones((self.CONSTANT_ROW_SIZE,
                                                 self.CONSTANT_COL_SIZE)))
                    elif np.ma.size(left_matrix, 0) > np.ma.size(left_output, 0):
                        left_output = np.kron(left_output,
                                              np.ones((self.CONSTANT_ROW_SIZE, self.CONSTANT_COL_SIZE)))

                left_matrix = np.add(left_matrix, left_output)

            if len(predecessors) == 0 and node > 0:
                # weight = self.network[primary][node]['weight']
                return self.node_output(node, left_matrix, left_matrix)

            # logit will be weighted sum of predecessor outputs
            for predecessor in predecessors:
                weight = self.network[predecessor][node]['weight']
                predecessor_output = weight * self.get_phenotype(predecessor)

                # rescale smaller matrix if need be.
                while np.ma.size(predecessor_output, 0) != np.ma.size(logit, 0):
                    if np.ma.size(logit, 0) < np.ma.size(predecessor_output, 0):
                        logit = np.kron(logit,
                                        np.ones((self.CONSTANT_ROW_SIZE,
                                                 self.CONSTANT_COL_SIZE)))
                    elif np.ma.size(logit, 0) > np.ma.size(predecessor_output, 0):
                        predecessor_output = np.kron(predecessor_output,
                                                     np.ones((self.CONSTANT_ROW_SIZE, self.CONSTANT_COL_SIZE)))

                logit = np.add(logit, predecessor_output)

            if len(left_nodes) == 0 and node > 0:
                # weight = self.network[primary][node]['weight']
                return self.node_output(node, logit, logit)

            return self.node_output(node, left_matrix, logit)

    def node_output(self, node, left_matrix, logit):

        if self.network.node[node]['type'] == 'kronecker':
            return np.kron(left_matrix, logit)


        # rescale smaller matrix if need be.
        while np.ma.size(left_matrix, 0) != np.ma.size(logit, 0):
            if np.ma.size(logit, 0) < np.ma.size(left_matrix, 0):
                logit = np.kron(logit,
                                np.ones((self.CONSTANT_ROW_SIZE,
                                         self.CONSTANT_COL_SIZE)))
            elif np.ma.size(logit, 0) > np.ma.size(left_matrix, 0):
                left_matrix = np.kron(left_matrix,
                                            np.ones((self.CONSTANT_ROW_SIZE,
                                                     self.CONSTANT_COL_SIZE)))

        if node == 0:
            # output node acts just like a sum
            return np.add(left_matrix, logit)
        elif self.network.node[node]['type'] == 'elem_mult':
            return np.multiply(left_matrix, logit)

    def left_nodes(self, node, predecessors):

        left_nodes = []

        if self.network.node[node]['type'] == 'constant':
            return left_nodes

        for predecessor in predecessors:
            if self.network[predecessor][node]['left']:
                left_nodes.append(predecessor)

        # if there is no primary return None
        return left_nodes


# Testing Genotype stuff

genome = Genome()

for i in range(0, 1):
    genome.mutate()

node_size = 1000
node_color = 'white'
node_text_size = 15
edge_alpha = 0.3
edge_thickness = 1
edge_text_pos = 0.8
text_font = 'sans-serif'

labels = {}
for i in range(nx.number_of_nodes(genome.network)):
    if genome.network.node[i]['type'] == 'constant':
        word = "\n" + str(genome.network.node[i]['constant'])
    else:
        word = " "
    labels[i] = str(genome.network.node[i]['type']) + word

colorList = []

for i in genome.network.edges():
    a, b = i
    colorVal = genome.network.edge[a][b]['weight']
    colorList.append(colorVal)

pos = nx.drawing.nx_agraph.graphviz_layout(genome.network, prog='dot')

nx.draw_networkx_nodes(genome.network, pos, node_size=node_size, node_color=node_color)
nx.draw_networkx_edges(genome.network, pos, width=colorList, edge_color=colorList, edge_cmap=plt.cm.RdYlGn, arrows=True)
nx.draw_networkx_labels(genome.network, pos, labels=labels, font_size=node_text_size,
                        font_family=text_font)
nx.draw_networkx_edge_labels(genome.network, pos)
plt.show()

print "getting phenotype"
print genome.get_phenotype()
# print np.size(genome.get_phenotype(), 0)

