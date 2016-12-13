import numpy as np
import networkx as nx
import random
from enum import Enum
import matplotlib.pyplot as plt
import tkinter
# import scipy.signal
# import pygraphviz
# import pydot

#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.

# If it is not possible or desirable to put the notice in a particular
# file, then You may include the notice in a location (such as a LICENSE
# file in a relevant directory) where a recipient would be likely to look
# for such a notice.

# You may add additional accurate notices of copyright ownership.

# Written by Stefano Palmieri in December 2016


# This class stores the genome as a weighted directed graph representing
# a Compositional Adjacency Matrix Producing Network. The weighted directed
#  graph is produced from Nodes and Links. Nodes can output a constant
# matrix (tensor) or produce the matrix product, transpose, convolution, or
# Kronecker product of the input matrices. Links are floating point values
# that are use for scalar multiplication of matrices outputted from Nodes.

# Enumerate node types
# If you want to add more node types, add after Hadamard
class NodeType(Enum):
    output = 0
    input = 1
    # actual matrix operations get placed under this line
    #############################################
    kronecker = 2
    hadamard = 3


class Genome(object):

    # Probability of mutating a new node
    P_NEW_NODE = 0.1

    # Probability of mutating a new input node
    P_NEW_INPUT_NODE = 0.1

    # Probability of mutating a new link
    P_NEW_LINK = 0.3

    # Probability of mutating an existing input node
    P_INPUT_NODE = 0.3

    # Probability of mutating existing link
    P_LINK = 0.3

    # Weights of a new added link
    N_LINK_WEIGHT = 1.0

    # Sizes for matrices in the input node type
    CONSTANT_ROW_SIZE = 4
    CONSTANT_COL_SIZE = 4

    # Mean and standard deviation for weight mutations
    MUTATE_MEAN = 0.1
    MUTATE_STD = 0.1

    def __init__(self):
        self.network = nx.MultiDiGraph()

        # node with 0 id is the output
        self.network.add_node(0, type=NodeType.output)
        self.network.add_node(1, type=NodeType.input,
                              constant=np.zeros((self.CONSTANT_ROW_SIZE, self.CONSTANT_COL_SIZE), dtype=np.float16))
        self.network.add_edge(1, 0, key=True, weight=1.0)

    # Mutations can add a Node, add a Link, change the weight of a Link,
    # change the function of a Node.
    def mutate(self):

        # If mutation occurs, mutate a new node
        if random.random() <= self.P_NEW_NODE:
            print("P_NEW_NODE")
            self.mutate_new_node()

        # If mutation occurs, mutate a new input node
        if random.random() <= self.P_NEW_INPUT_NODE:
            print("P_NEW_INPUT_NODE")
            self.mutate_new_input_node()

        # If mutation occurs, mutate an existing input node
        if random.random() <= self.P_INPUT_NODE:
            print("P_INPUT_NODE")
            self.mutate_node()

        # If mutation occurs, mutate a new link
        if random.random() <= self.P_NEW_LINK:
            self.mutate_new_link()

        # If mutation occurs, mutate an existing link
        if random.random() <= self.P_LINK:
            print("P_LINK")
            self.mutate_link()

    # Create a new random Node by selecting a link and splitting it
    def mutate_new_node(self):

        # Randomly select a link from the existing network
        start, end, key = random.choice(self.network.edges(keys=True))

        # Randomly select the function for this node but don't include Input or Output types as functions
        function = NodeType(random.randrange(NodeType.input.value + 1, len(NodeType)))

        # Create a new node with the function type
        node_num = self.network.number_of_nodes()
        self.network.add_node(node_num, type=function)

        # Add links for the new node
        self.network.add_edge(start, node_num, key=random.random() < 0.5, weight=self.N_LINK_WEIGHT)
        self.network.add_edge(node_num, end, key=key, weight=self.N_LINK_WEIGHT)

        # Remove old edge (in future versions this edge should be disabled rather than removed)
        self.network.remove_edge(start, end, key=key)

    # Create a new random Node by selecting a non-input node and creating a child
    def mutate_new_input_node(self):

        # Get list of non-input nodes
        non_input_nodes = [node for node, data in self.network.nodes(data=True) if data['type'] != NodeType.input]

        # randomly pick a node from the list
        node = random.choice(non_input_nodes)

        # Create a new input node
        node_num = self.network.number_of_nodes()
        self.network.add_node(node_num, type=NodeType.input,
                              constant=np.zeros((self.CONSTANT_ROW_SIZE, self.CONSTANT_COL_SIZE), dtype=np.float16))

        # Create a link from the new input node to existing non-input node
        self.network.add_edge(node_num, node, key=random.random() < 0.5, weight=self.N_LINK_WEIGHT)

    # Randomly mutates one of the existing input nodes in the CAMPN.
    def mutate_node(self):

        # get list of input nodes
        input_nodes = [node for node, data in self.network.nodes(data=True) if data['type'] == NodeType.input]

        # randomly pick a node from the list
        node = random.choice(input_nodes)

        # mutate one of it's constant's elements
        rand_row = random.randrange(0, self.CONSTANT_ROW_SIZE)
        rand_col = random.randrange(0, self.CONSTANT_COL_SIZE)

        # Mutate by Gaussian random variable
        change = np.random.normal(self.MUTATE_MEAN, self.MUTATE_STD)

        # apply the change
        self.network.node[node]['constant'][rand_row, rand_col] += change

    # Create a new Link and add it to the Genome.
    def mutate_new_link(self):
        # select a random node to start a link from and, don't include output node
        list = [node for node, data in self.network.nodes(data=True) if data['type'] != NodeType.output]
        start = random.choice(list)

        # select a random node to end link at, don't include input nodes
        list = [node for node, data in self.network.nodes(data=True) if data['type'] != NodeType.input and node != start]
        end = random.choice(list)

        # Create the link
        key = random.random() < 0.5
        self.network.add_edge(start, end, key=key, weight=self.N_LINK_WEIGHT)

        # Check for no cycles. If there are, remove this link
        try:
            nx.find_cycle(self.network, 1)
            self.network.remove_edge(start, end, key=key)
        except nx.exception.NetworkXNoCycle:
            return

    # Randomly mutate an existing link by modifying the weight
    def mutate_link(self):

        # Select a random link
        start, end, key = random.choice(self.network.edges(keys=True))

        # Add a Gaussian random variable to existing weight
        weight_change = np.random.normal(self.MUTATE_MEAN, self.MUTATE_STD)

        # Modify the weight on the link
        self.network[start][end][key]['weight'] += weight_change

    # recursively compute output for a node
    def get_phenotype(self, node=None):

        if node is None:
            return self.get_phenotype(0)

        if self.network.node[node]['type'] == NodeType.input:
            return self.network.node[node]['constant']

        left_logit = self.calculate_logit(node, True)
        right_logit = self.calculate_logit(node, False)

        return self.node_output(node, left_logit, right_logit)

    def calculate_logit(self, node, key):
        predecessors = list(self.network.predecessors(node))
        nodes = [predecessor for predecessor in predecessors if key in self.network[predecessor][node]]

        if nodes:
            logit = np.zeros((self.CONSTANT_ROW_SIZE, self.CONSTANT_COL_SIZE))
            for x in nodes:
                weight = self.network[x][node][key]['weight']
                term = self.get_phenotype(x)
                logit, term = self.same_size(logit, term)
                logit += weight * term
            return logit
        # if there are no links for this logit, return false
        else:
            return None

    # compute output for the node
    def node_output(self, node, left_logit, right_logit):

        # output node acts just like a sum
        if self.network.node[node]['type'] == NodeType.output:
            if left_logit is None:
                return right_logit
            elif right_logit is None:
                return left_logit
            else:
                left_logit, right_logit = self.same_size(left_logit, right_logit)
                return np.add(left_logit, right_logit)

        elif self.network.node[node]['type'] == NodeType.input:
            return self.network.node[node]['constant']

        elif self.network.node[node]['type'] == NodeType.kronecker:
            if left_logit is None:
                result = np.kron(right_logit, right_logit)
                return result
            elif right_logit is None:
                result = np.kron(left_logit, left_logit)
                return result
            else:
                return np.kron(left_logit, right_logit)

        elif self.network.node[node]['type'] == NodeType.hadamard:
            if left_logit is None:
                return right_logit
            elif right_logit is None:
                return left_logit
            else:
                left_logit, right_logit = self.same_size(left_logit, right_logit)
                return np.multiply(left_logit, right_logit)

    # returns left and right so that they are the same size, assuming both are square
    # and share some common factor of their sides
    def same_size(self, left, right):
        left_row_size = np.ma.size(left, 0)
        right_row_size = np.ma.size(right, 0)
        if left_row_size != right_row_size:
            if left_row_size < right_row_size:
                length = int(right_row_size / left_row_size)
                left = np.kron(left, np.ones((length, length)))
            else:
                length = int(left_row_size / right_row_size)
                right = np.kron(right, np.ones((length, length)))

        return left, right

# Testing Genotype stuff

genome = Genome()

for i in range(0, 40):
    genome.mutate()

print("printing nodes")
for node in genome.network.nodes(data=True):
    print(node)

print("printing edges")
for edge in genome.network.edges(data=True, keys=True):
    print(edge)

node_size = 1000
node_color = 'white'
node_text_size = 15
edge_alpha = 0.3
edge_thickness = 1
edge_text_pos = 0.8
text_font = 'sans-serif'

pos = nx.spring_layout(genome.network)

nx.draw_networkx(genome.network, pos, node_size=node_size, node_color=node_color)
#nx.draw_networkx_nodes(genome.network, pos, node_size=node_size, node_color=node_color)
#nx.draw_networkx_edges(genome.network, pos, width=colorList, edge_color=colorList, edge_cmap=plt.cm.RdYlGn, arrows=True)
#nx.draw_networkx_labels(genome.network, pos, labels=labels, font_size=node_text_size,
#                        font_family=text_font)
#nx.draw_networkx_edge_labels(genome.network, pos)
plt.show()

print("getting phenotype")
phenotype = genome.get_phenotype()
print(phenotype)
print(np.size(phenotype, 0))

plt.matshow(phenotype, fignum=100, cmap=plt.cm.gray)
plt.show()