import networkx as nx
import matplotlib.pyplot as plt
import pdb
import nodes
import pydot
import random
from graphviz import Digraph
from analyzer import Q


def node_display_name(node):
    return node.__class__.__name__ + '\n ' + node.ID


def visit_tree(pointer, vertices, edges):
    vertices.append((str(pointer), pointer.__class__.__name__))
    for i in range(3):
        #if (not isinstance(pointer, nodes.Cell)):
        if (not(type(pointer.child[i]) in {nodes.Cell, nodes.Conns, type(None)})):
            edges.append((
                str(pointer),
                str(pointer.child[i])
            ))
            visit_tree(pointer.child[i], vertices, edges)


class IndividualVisualizer():

    def __init__(self, individual):
        self.individual = individual
        self.individual.execute()

    def draw_tree(self, loc='test_tree.pdf'):
        pointer = self.individual.child[0]
        vertices = []
        edges = []
        visit_tree(pointer, vertices, edges)
        # pdb.set_trace()
        g = pydot.Dot(graph_type='graph', ranksep='0.5 equally', nodesep=0.2)
        g.set_node_defaults(shape='ellipse')
        for v in vertices:
            g.add_node(pydot.Node(v[0], label=v[1]))
        for e in edges:
            g.add_edge(pydot.Edge(e[0], e[1]))
        g.write(loc, format='pdf')

    def draw_modular(self, loc='test_mod.pdf', one_cut=False):
        nn = self.individual.nn
        Qscore, groups = Q(nn, one_cut)
        g = pydot.Dot(graph_type='digraph', ranksep='0.5 equally', nodesep=0.2)
        g.set_node_defaults(shape='circle',style='filled')
        inputs = pydot.Subgraph('inputs',graph_type='digraph',rank='source')
        outputs = pydot.Subgraph('outputs',graph_type='digraph',rank='same')
        g.add_subgraph(inputs)
        g.add_subgraph(outputs)
        for i in range(nn.input_size):
            inputs.add_node(pydot.Node(str(i), label='input'+str(i)))
            if (i > 0):
                inputs.add_edge(pydot.Edge(str(i - 1), str(i), style='invis'))
        for i in range(nn.input_size, nn.input_size + nn.output_size):
            outputs.add_node(pydot.Node(str(i), label='output'+str(i - nn.input_size)))
            if (i > nn.input_size):
                outputs.add_edge(pydot.Edge(str(i - 1), str(i), style='invis'))
        count = -1
        for group in groups:
            count += 1
            random_number = random.randint(0,16777215)
            hex_color = str(hex(random_number))
            hex_color = '#' + hex_color[2:]
            cluster = pydot.Cluster('c' + str(count))
            for n in group:
                if (n >= nn.input_size + nn.output_size):
                    cluster.add_node(pydot.Node(str(n), fillcolor= hex_color))
            g.add_subgraph(cluster)
        edges = tuple(zip(nn.connList.connSource, nn.connList.connTarget))
        for e in edges:
            g.add_edge(pydot.Edge(str(e[0]), str(e[1])))
        g.write(loc, format='pdf')

    def draw_nn(self, loc='test_nn.png', drawWeight=False):
        self.individual.nn.compile()
        vertices = list(range(self.individual.nn.nodeCount))
        clist = self.individual.nn.connList
        nnGraph = nx.MultiDiGraph()
        nnGraph.add_nodes_from(vertices)
        if drawWeight:
            edges = tuple(zip(clist.connSource, clist.connTarget, clist.connWeight))
            nnGraph.add_weighted_edges_from(edges)
        else:
            edges = tuple(zip(clist.connSource, clist.connTarget))
            nnGraph.add_edges_from(edges)
        layout = nx.nx_agraph.graphviz_layout(
            nnGraph,
            prog='dot',
            args='-Gnodesep=5')
        plt.figure(figsize=(20, 20))
        nx.draw(nnGraph, layout, with_labels=True, node_size=1000)
        labels = nx.get_edge_attributes(nnGraph, 'weight')
        nx.draw_networkx_edge_labels(nnGraph, layout, edge_labels=labels)
        plt.savefig(loc)
