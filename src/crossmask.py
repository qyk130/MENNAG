import numpy as np
import random
from nn.FC import FC
from analyzer import Q
import igraph as ig
import pandas as pd
#import leidenalg


def tree_to_fos(group, max_depth, depth):

    def traverse(tree):
        group = []
        if (isinstance(tree[0], int)):
            return tree
        group.extend(traverse(tree[0]))
        group.extend(traverse(tree[1]))
        return group

    if (isinstance(group[0], int)):
        return [group]

    try:
        a = group[0] + group[1]
    except IndexError:
        print(group)

    if (depth != max_depth):
        fos = []
        fos_l = tree_to_fos(group[0], max_depth, depth + 1)
        fos_r = tree_to_fos(group[1], max_depth, depth + 1)
        fos.extend(fos_l)
        fos.extend(fos_r)
        new_group = set()
        for s in fos_l:
            new_group = new_group.union(s)
        for s in fos_r:
            new_group = new_group.union(s)
        fos.append(list(new_group))
        return fos
    else:
        return [traverse(group)]


def modular_cross_mask(max_depth, layer_size, weights, one_cut):
    #does not include output bias
    def get_outputs_pos(layer_size, i, j):
        pos = 0
        #bias
        for l in range(1, len(layer_size)):
            pos += layer_size[l]
        for l in range(0, i):
            pos += layer_size[l] * layer_size[l + 1]
        pos += layer_size[i + 1] * j
        return range(pos, pos + layer_size[i + 1])

    output_bias_size = layer_size[-1]
    Asum = np.zeros((len(weights[0])-1, len(weights[0])-1))
    for n in range(len(weights)):
        A = np.zeros((len(weights[0])-1, len(weights[0])-1))
        edge_pos = 0
        # bias
        for i in range(1, len(layer_size) - 1):
            for j in range(layer_size[i]):
                outputs_pos = get_outputs_pos(layer_size, i, j)
                for k in outputs_pos:
                    A[edge_pos, k-output_bias_size] = abs(weights[n][edge_pos] * weights[n][k])
                    A[k-output_bias_size, edge_pos] = A[edge_pos, k-output_bias_size]
                edge_pos += 1

        # weights
        for i in range(1, len(layer_size) - 1):
            outputs_pos = []
            for k in range(layer_size[i + 1]):
                outputs_pos.append(get_outputs_pos(layer_size, i, k))
            for j in range(layer_size[i]):
                for k in range(layer_size[i + 1]):
                    for l in outputs_pos[k]:
                        A[edge_pos, l-output_bias_size] = abs(weights[n][edge_pos + output_bias_size] * weights[n][l])
                        A[l-output_bias_size, edge_pos] = A[edge_pos, l-output_bias_size]
                    edge_pos += 1
        A = A / np.sum(A)
        Asum = Asum + A
    Asum = Asum / len(weights)
    q_value, group = Q(Asum, 0.5, one_cut)
    #print(q_value, len(groups[0]), len(groups[1]))
    #print(len(group1), len(group2))
    groups = tree_to_fos(group, max_depth, 0)
    bias_pos = 0
    for i in range(len(layer_size) - 1):
        bias_pos += layer_size[i]
    new_groups = []
    for g in groups:
        new_g = []
        for i in g:
            if i > bias_pos:
                new_g.append(i + output_bias_size)
        new_groups.append(new_g)
    #print(q_value)
    return new_groups, q_value

def leiden_cross_mask(max_depth, layer_size, weights, one_cut):
    #does not include output bias
    def get_outputs_pos(layer_size, i, j):
        pos = 0
        #bias
        for l in range(1, len(layer_size)):
            pos += layer_size[l]
        for l in range(0, i):
            pos += layer_size[l] * layer_size[l + 1]
        pos += layer_size[i + 1] * j
        return range(pos, pos + layer_size[i + 1])

    size = 0
    for i in range(0, len(layer_size) - 2):
        size += (layer_size[i] + 1) * layer_size[i + 1] * layer_size[i + 2]
    a = np.zeros((size,3))
    df = pd.DataFrame(a,columns=range(3))
    output_bias_size = layer_size[-1]
    #g = ig.Graph()
    #g.add_vertices(len(weights[0])-1)
    for n in range(len(weights)):
        edge_pos = 0
        graph_edge_count = 0
        # bias
        for i in range(1, len(layer_size) - 1):
            for j in range(layer_size[i]):
                outputs_pos = get_outputs_pos(layer_size, i, j)
                for k in outputs_pos:
                    proximity = float(abs(weights[n][edge_pos] * weights[n][k]))
                    if (n == 0):
                        df.iloc[graph_edge_count] = (edge_pos, k-output_bias_size, proximity)
                        #g.add_edges([(edge_pos, k-output_bias_size)])
                        #g.es.find(_source=edge_pos, _target=k-output_bias_size)['weight'] = \
                        #        float(abs(weights[n][edge_pos] * weights[n][k]))
                    else:
                        df.iloc[graph_edge_count,2] += proximity
                        #g.es.find(_source=edge_pos, _target=k-output_bias_size)['weight'] += \
                        #        float(abs(weights[n][edge_pos] * weights[n][k]))
                    graph_edge_count += 1
                edge_pos += 1
        # weights
        for i in range(0, len(layer_size) - 2):
            outputs_pos = []
            for k in range(layer_size[i + 1]):
                outputs_pos.append(get_outputs_pos(layer_size, i + 1, k))
            for j in range(layer_size[i]):
                for k in range(layer_size[i + 1]):
                    for l in outputs_pos[k]:
                        proximity = float(abs(weights[n][edge_pos + output_bias_size] * weights[n][l]))
                        #if proximity > 0:
                        #    print(edge_pos + output_bias_size, l)
                        if (n == 0):
                            #if (edge_pos == 24):
                            #    print(edge_pos ,l-output_bias_size)
                            df.iloc[graph_edge_count] = (edge_pos, l-output_bias_size, proximity)
                            #g.add_edges([(edge_pos, l-output_bias_size)])
                            #g.es.find(_source=edge_pos, _target=l-output_bias_size)['weight'] = \
                            #        float(abs(weights[n][edge_pos + output_bias_size] * weights[n][l]))
                        else:
                            df.iloc[graph_edge_count, 2] += proximity
                            #g.es.find(_source=edge_pos, _target=l-output_bias_size)['weight'] += \
                            #        float(abs(weights[n][edge_pos + output_bias_size] * weights[n][l]))
                        graph_edge_count += 1
                    edge_pos += 1
    tuple_list = list(df.itertuples(index=False, name=None))
    g = ig.Graph.TupleList(tuple_list, weights=True)
    #for i in range(len(g.es)):
    #    if g.es['weight'][i] is None:
    #        print(g.es[i].source, g.es[i].target)
    #print(list(g.es[:]))
    part = g.community_leiden(objective_function='modularity', weights='weight')
    groups = [part[i] for i in range(len(part))]
    #groups = [list(range(len(weights)))]
    #groups.extend(group)
    #print(len(groups))
    #print(sum([len(s) for s in groups]))
    '''
    for i in range(2**max_depth):
        b = np.random.randint(0, 2, len(group))
        g1 = []
        for j in range(len(group)):
            if b[j] == 0:
                g1.extend(group[j])
        groups.append(g1)
    '''
    bias_pos = 0
    for i in range(1,len(layer_size) - 1):
        bias_pos += layer_size[i]
    new_groups = []
    for group in groups:
        new_group = []
        for i in group:
            if g.vs[i]['name'] >= bias_pos:
                new_group.append(int(g.vs[i]['name'] + output_bias_size))
            else:
                new_group.append(int(g.vs[i]['name']))
        new_groups.append(new_group)
    #print(new_groups)
    #print(part.modularity)
    return new_groups, part.modularity
