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
    P_NEW_NODE = 0.09

    # Probability of mutating a new Link
    P_NEW_LINK = 0.1

    # Probability of mutating existing node
    P_NODE = 0.1

    # Probability of mutating existing link
    P_LINK = 0.5

    # Weights of a new added link
    N_LINK_WEIGHT = 0.1

    # Dictionary for types of Nodes
    types = {
        0: 'constant',
        1: 'matrix_prod',
        2: 'transpose',
        3: 'kronecker',
        # convolution not included
        # 4: 'convolution'
        # element-wise multiply not included
        # 5: 'elem_mult'
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
        self.network.add_node(0, type='constant',
                              constant=np.matrix([[0.0, 1.0], [0.0, 0.0]], np.float16))
        self.network.add_node(1, type='kronecker',
                              constant=np.matrix([[1.0, 0.0], [0.0, 1.0]], np.float16))
        self.network.add_edge(1, 0, weight=1.0, primary=True)
        self.network.add_node(2, type='constant',
                              constant=np.matrix([[0.0, 0.0], [0.0, 0.0]], np.float16))
        self.network.add_edge(2, 1, weight=1.0, primary=True)
        self.network.add_node(3, type='constant',
                              constant=np.matrix([[0.0, 0.0], [0.0, 0.0]], np.float16))
        self.network.add_edge(3, 1, weight=1.0, primary=False)

    # Mutations can add a Node, add a Link, change the weight of a Link,
    # change the function of a Node.
    def mutate(self):
        # If mutation occurs, mutate a new node
        rand = random.random()
        if rand <= self.P_NEW_NODE:
            self.mutate_new_node()
        # If mutation occurs, mutate a new link
        rand = random.random()
        if rand <= self.P_NEW_LINK:
            self.mutate_new_link()
        # If mutation occurs, mutate an existing node
        rand = random.random()
        if rand <= self.P_NODE:
            self.mutate_node()
        # If mutation occurs, mutate an existing link
        rand = random.random()
        if rand <= self.P_LINK:
            self.mutate_link()

    # Randomly mutates one of the existing nodes in the CAMPN. A Node can
    # be mutated by changing the function that the node performs.
    def mutate_node(self):
        # pick a random node that's not the output node
        node = random.randrange(1, self.network.number_of_nodes())

        # pick a type of function to mutate to
        func = self.types[random.randrange(0, len(self.types))]

        self.network.node[node]['type'] = func

        # mutate one of it's constant's elements
        constant = self.network.node[node]['constant']
        rand_row = random.randrange(0, self.CONSTANT_ROW_SIZE)
        rand_col = random.randrange(0, self.CONSTANT_COL_SIZE)

        # Mutate by Gaussian random variable
        change = np.random.normal(self.MUTATE_MEAN, self.MUTATE_STD)

        # apply the change
        constant[rand_row, rand_col] += change
        self.network.node[node]['constant'] = constant

    # Randomly mutate an existing link by modifying the weight or by
    # changing the Node that one of the outputs connects to.
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
        func = self.types[random.randrange(0, len(self.types))]

        # Randomly select a node from the existing network
        end = random.randrange(0, self.network.number_of_nodes()-1) + 1

        matrix = np.random.rand(self.CONSTANT_ROW_SIZE, self.CONSTANT_COL_SIZE).astype(np.float16)

        # add the node to the network and link it
        self.network.add_node(self.network.number_of_nodes(),
                              type=func, constant=matrix)
        self.network.add_edge(self.network.number_of_nodes()-1,
                              end, weight=self.N_LINK_WEIGHT,
                              primary=len(self.network.predecessors(end)) == 0)

    # Create a new Link and add it to the Genome.
    def mutate_new_link(self):
        # select a random node to start a link from, don't include output node
        start = random.randrange(1, self.network.number_of_nodes())

        # Keep picking a terminal node until one that is not the
        # start link is found
        while True:
            end = random.randrange(0, self.network.number_of_nodes())
            if end != start:
                break

        # First check this link doesn't already exist so we don't add it again.
        # Otherwise we can consider this mutations as failed if it does exist
        # (no mutation occurred).
        if not (start, end) in self.network.edges():
            # If the end node has no incoming links, make link primary.
            self.network.add_edge(start, end, weight=self.N_LINK_WEIGHT,
                                  primary=len(self.network.predecessors(end)) == 0)

            # attempt to add the link. Do this checking that
            # the number of simple cycles does not become one or greater.
            # Otherwise we have to remove the link to keep the CTPPN acyclic
            # and can consider this a mutation that failed (no mutation occurred).
            cycles = len(list(nx.simple_cycles(self.network)))

            if cycles >= 1:
                self.network.remove_edge(start, end)

    # recursively compute output for a node
    def get_phenotype(self, node=None):

        if node is None:
            return self.get_phenotype(0)

        else:
            # optimization for constant node so don't have to calculate sub
            # phenotype
            if self.network.node[node]['type'] == 'constant' and node != 0:
                return self.network.node[node]['constant']

            predecessors = list(self.network.predecessors(node))
            primary = self.primary_edge_node(node, predecessors)

            logit = np.zeros((self.CONSTANT_ROW_SIZE, self.CONSTANT_COL_SIZE))

            # if nothing is connected return it's constant matrix
            if primary is None:
                return self.network.node[node]['constant']

            # remove the primary from the predecessor list
            predecessors.remove(primary)

            # if there are no other predecessors other than the primary
            # return the primary's phenotype
            if len(predecessors) == 0:
                weight = self.network[primary][node]['weight']
                return weight * self.get_phenotype(primary)

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

            return self.node_output(node, primary, logit)

    def node_output(self, node, primary, logit):
        # weight of the primary link
        weight = self.network[primary][node]['weight']

        primary_phenotype = weight * self.get_phenotype(primary)

        # rescale smaller matrix if need be.
        while np.ma.size(primary_phenotype, 0) != np.ma.size(logit, 0):
            if np.ma.size(logit, 0) < np.ma.size(primary_phenotype, 0):
                logit = np.kron(logit,
                                np.ones((self.CONSTANT_ROW_SIZE,
                                         self.CONSTANT_COL_SIZE)))
            elif np.ma.size(logit, 0) > np.ma.size(primary_phenotype, 0):
                primary_phenotype = np.kron(primary_phenotype,
                                            np.ones((self.CONSTANT_ROW_SIZE,
                                                     self.CONSTANT_COL_SIZE)))

        if node == 0:
            # output node acts just like a sum
            return np.add(primary_phenotype, logit)
        elif self.network.node[node]['type'] == 'matrix_prod':
            return primary_phenotype * logit
        elif self.network.node[node]['type'] == 'transpose':
            return np.transpose(np.add(primary_phenotype, logit))
        elif self.network.node[node]['type'] == 'kronecker':
            return np.kron(primary_phenotype, logit)

    def primary_edge_node(self, node, predecessors):

        for predecessor in predecessors:
            if self.network[predecessor][node]['primary']:
                return predecessor

        # if there is no primary return None
        return None


# Testing Genotype stuff

genome = Genome()

for i in range(0, 2):
    genome.mutate()


node_size=1000
node_color='white'
node_text_size=15
edge_alpha=0.3
edge_thickness=1
edge_text_pos=0.8
text_font='sans-serif'

labels = {}
for i in range(nx.number_of_nodes(genome.network)):
    labels[i] = str(genome.network.node[i]['type']) # + "\n" + str(genome.network.node[i]['constant'])

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
#nx.draw_networkx_edge_labels(genome.network, pos)
#plt.show()

#print genome.get_phenotype()

