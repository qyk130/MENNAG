import random
import numpy as np
import configs
import activations
from nn.feedforward import FeedForward
from model import Model, min_common_ancestry_dist
import pdb

class Root(Model):

    def __init__(self, config=None, ID=0, birth_gen=-1):
        super().__init__(config, ID, birth_gen=birth_gen)
        self.depth = -1
        self.fitness = 0

    def generate(self):
        self.child = [Div(self), None, None]
        self.child[0].generate()

    def compile(self):
        self.child[0].ID = ''
        self.maxDepth = self.child[0].update_depth()
        self.child[0].compile()
        self.neuronSet = self.child[0].neuronSet
        self.connSet = self.child[0].connSet
        self.inSet = self.child[0].inSet
        self.outSet = self.child[0].outSet
        self.nodeSet = self.child[0].nodeSet

    def clear_sets(self):
        for c in self.child:
            if (c is not None):
                c.reset()
                c.clear_sets()
        return

    def execute(self):
        self.nn = FeedForward(self.config)
        for key, value in self.neuronSet.items():
            self.nn.add_node(key, value)
        for key, value in self.connSet.items():
            source = key[0]
            target = key[1]
            try:
                while (source not in self.neuronSet.keys() and source != ''):
                    source = source[:len(source) - 1]
                while (target not in self.neuronSet.keys() and target != ''):
                    target = target[:len(target) - 1]
            except IndexError:
                print("IndexError in neuron id reduction")
            if (not(source == '' or target == '')):
                #self.nn.add_node(source, self.neuronSet[source])
                #self.nn.add_node(target, self.neuronSet[target])
                self.nn.add_conn(source, target, value)
        for inConn in self.inSet:
            #self.nn.add_node(inConn[1], self.neuronSet[inConn[1]])
            self.nn.add_conn('i' + str(inConn[0]), inConn[1], inConn[2])
        for outConn in self.outSet:
            #self.nn.add_node(outConn[0], self.neuronSet[outConn[0]])
            self.nn.add_conn(outConn[0], 'o' + str(outConn[1]), outConn[2])
        try:
            self.nn.add_finish()
        except IndexError:
            print("indexerror in add_finish")
        self.clear_sets()
        return self.nn

    def detach(self):
        self.nn = None

    def mutate(self):
        self.trajectory.append('mutation')
        n0 = random.random()
        insertionThreshold = self.config.insertion_rate
        deletionThreshold = insertionThreshold + self.config.deletion_rate
        if (n0 < insertionThreshold):
            n1 = random.random()
            if (n1 < 0.5):
                type = np.random.choice([Div, Clones], p=[0.8, 0.2])
                self.insert_at_div(type)
            else:
                type = np.random.choice([Conn, Clone], p=[0.8, 0.2])
                self.insert_at_list(type)
        elif (n0 < deletionThreshold):
            n1 = random.random()
            if (n1 < 0.5):
                type = np.random.choice([Div, Clones], p=[0.8, 0.2])
                self.delete_at_div(type)
            else:
                type = np.random.choice([Conn, Clone], p=[0.8, 0.2])
                self.delete_at_list(type)
        else:
            self.child[0].mutate()
        if (self.config.ea == "speciation"):
            self.add_ancestors()

    def get_node(self, ID):
        pointer = self.child[0]
        for i in range(len(ID)):
            pointer = pointer.child[int(ID[i])]
            if (pointer is None):
                return None
        return pointer

    def get_all_nodes(self, type=None, rule=None, depth=None, maxDepth=None):
        candidates = []
        for n in self.nodeSet:
            if ((type is None) or (isinstance(n, type))):
                if ((rule is None) or (n.rule == rule)):
                    if ((depth is None) or (n.depth == depth)):
                        if ((maxDepth is None) or (n.maxDepth <= maxDepth)):
                            candidates.append(n)
        return candidates

    def insert_at_div(self, type):
        candidates = []
        availRules = [0, 1, 2]
        while((len(candidates) < 1) and (len(availRules) > 0)):
            selectRule = random.choice(availRules)
            candidates = self.get_all_nodes(
                type=Div,
                rule=selectRule,
                maxDepth=self.config.max_depth - 1
                )
            availRules.remove(selectRule)
        if (len(candidates) == 0):
            return 0
        insertAt = random.choice(candidates)
        if (selectRule == 1):
            insertPos = 0
        if (selectRule == 2):
            insertPos = 1
        elif (selectRule == 0):
            insertPos = random.choice([0, 1])
        n1 = insertAt.child[insertPos]
        n2 = Div(insertAt)
        if (type == Div):
            n2.rule = 0
            newPos = random.choice([0, 1])
            oldPos = 1 - newPos
        else:
            n2.rule = random.choice([1, 2])
            if (n2.rule == 1):
                newPos = 1
                oldPos = 0
            else:
                newPos = 0
                oldPos = 1
        n2.child[oldPos] = n1
        n1.parent = n2
        n2.child[newPos] = type(n2)
        n2.child[newPos].generate()
        n2.child[2] = Conns(n2)
        n2.child[2].generate()
        insertAt.child[insertPos] = n2

    def delete_at_div(self, type):
        candidates = []
        if (type == Div):
            selectRule = 0
        elif (type == Clones):
            selectRule = random.choice([1, 2])
        candidates = self.get_all_nodes(type=Div, rule=selectRule)
        if (len(candidates) == 0):
            return 0
        selected = random.choice(candidates)
        if (selected == self.child[0]):
            return
        selectedPos = 0
        for i in range(3):
            if (selected.parent.child[i] == selected):
                selectedPos = i
        if (selectRule == 0):
            deletePos = random.choice([0, 1])
        elif (selectRule == 1):
            deletePos = 1
        elif (selectRule == 2):
            deletePos = 0
        siblingPos = 1 - deletePos
        deleted = selected.child[deletePos]
        sibling = selected.child[siblingPos]
        sibling.parent = selected.parent
        sibling.parent.child[selectedPos] = sibling
        return 1

    def insert_at_list(self, type):
        if (type == Clone):
            listType = Clones
            maxDepth = self.config.max_depth - 1
        elif (type == Conn):
            listType = Conns
            maxDepth = None
        candidates = self.get_all_nodes(type=listType, maxDepth = maxDepth)
        if (len(candidates) == 0):
            return
        insertAt = random.choice(candidates)
        listNode = listType(insertAt.parent)
        try:
            insertAt.parent.child[insertAt.get_pos()] = listNode
        except TypeError:
            pdb.set_trace()
            print("insert_at_list type error:", deleted)
        insertAt.parent = listNode
        listNode.child[1] = insertAt
        listNode.child[0] = type(listNode)
        listNode.child[0].generate()
        listNode.rule = 0

    def delete_at_list(self, type):
        if (type == Clone):
            listType = Clones
        elif (type == Conn):
            listType = Conns
        candidates = self.get_all_nodes(type=listType, rule=0)
        if (len(candidates) == 0):
            return 0
        deleted = random.choice(candidates)
        try:
            deleted.parent.child[deleted.get_pos()] = deleted.child[1]
            deleted.child[1].parent = deleted.parent
        except TypeError:
            pdb.set_trace()
            print("delete_at_list type error:", deleted)
        return 1
        
    def cross_with(self, p2):
        offspring = self.deepcopy()
        offspring.parents = [self.ID, p2.ID]
        p1nodes = offspring.get_all_nodes(type=Div)
        p1nodes.remove(offspring.child[0])
        random_weights = np.array([1/n.depth for n in p1nodes])
        random_weights /= sum(random_weights)
        p2nodes = []
        zero = False
        while(len(p2nodes) == 0):
            zero = True
            div1 = np.random.choice(p1nodes, p=random_weights)
            p2nodes = p2.get_all_nodes(type=Div, depth=min(div1.depth, p2.maxDepth))
        div2 = random.choice(p2nodes).deepcopy(div1.parent)
        if (div1.parent.child[0] == div1):
            div1.parent.child[0] = div2
        else:
            div1.parent.child[1] = div2
        offspring.trajectory.append('crossover')
        if (self.config.ea == 'speciation'):
            offspring.add_ancestors(p2.ancestors)
        return offspring

    def deepcopy(self):
        copy = Root(config=self.config)
        copy.trajectory = self.trajectory.copy()
        copy.ancestors = self.ancestors.copy()
        copy.child = [self.child[0].deepcopy(copy), None, None]
        copy.compile()
        return copy


class TreeNode():

    def __init__(self, parent):
        self.parent = parent
        self.depth = parent.depth
        self.config = parent.config
        self.child = [None, None, None]
        self.rule = -1
        self.ID = ''
        self.generated = False
        self.reset()

    def reset(self):
        self.neuronSet = {}
        self.connSet = {}
        self.inSet = set()
        self.outSet = set()
        self.nodeSet = set([self])

    def clear_sets(self):
        for c in self.child:
            if (c is not None):
                c.reset()
                c.clear_sets()
        return

    def mutate(self):
        for c in self.child:
            if (c is not None):
                c.mutate()
        return

    def generate(self):
        self.generated = True
        for c in self.child:
            if (c is not None):
                c.generate()
        return

    def compile(self):
        self.reset()
        for c in self.child:
            if (c is not None):
                c.compile()
        self.merge_from_children()
        return

    def update_depth(self):
        self.depth = self.parent.depth
        self.maxDepth = self.depth
        for c in self.child:
            if (c is not None):
                self.maxDepth = max(c.update_depth(), self.maxDepth)
        return self.maxDepth

    def merge_from_children(self):
        for c in self.child:
            if (c is not None):
                self.neuronSet.update(c.neuronSet)
                self.connSet.update(c.connSet)
                self.outSet.update(c.outSet)
                self.inSet.update(c.inSet)
                self.nodeSet.update(c.nodeSet)

    def get_pos(self):
        for i in range(3):
            if (self.parent.child[i] == self):
                return i

    def deepcopy(self, newParent):
        copy = self.__class__(newParent)
        copy.rule = self.rule
        copy.depth = self.depth
        copy.ID = self.ID
        for i in range(3):
            if (self.child[i] is not None):
                copy.child[i] = self.child[i].deepcopy(copy)
        return copy

class WeightedNode(TreeNode):

    def __init__(self, parent):
        super().__init__(parent)

    def weight_gen(self):
        mean = self.config.weight_mean
        std = self.config.weight_std
        self.weight = np.random.normal(mean, std)

    def weight_mut(self):
        number = random.random()
        resetThreshold = self.config.weight_reset_rate
        perturbThreshold = resetThreshold + self.config.weight_perturb_rate
        if (number < resetThreshold):
            self.generate()
        elif (number < perturbThreshold):
            self.weight += np.random.normal(0, self.config.perturb_std)


class Div(TreeNode):

    def __init__(self, parent):
        super().__init__(parent)
        self.depth += 1

    def generate(self):
        if (self.depth >= self.config.max_depth):
            # Rule #3 DIV -> CELL CELL
            self.rule = 3
        else:
            if (random.random() < 0.05):
                if (random.random() < 0.5):
                    # Rule #1 DIV -> DIV CLONES CONNS
                    self.rule = 1
                else:
                    # Rule #2 DIV -> CLONES DIV CONNS
                    self.rule = 2
            elif (random.random() < 1 / (self.depth**2 + 1e-5)):
                # Rule #0 DIV -> DIV DIV CONNS
                self.rule = 0
            else:
                # Rule #3 DIV -> CELL CELL
                self.rule = 3
        self.generate_by_rule()
        super().generate()

    def generate_by_rule(self):
        if (self.rule == 0):
            self.child[0] = Div(self)
            self.child[1] = Div(self)
            self.child[2] = Conns(self)
        elif (self.rule == 1):
            self.child[0] = Div(self)
            self.child[1] = Clones(self)
            self.child[2] = Conns(self)
        elif (self.rule == 2):
            self.child[1] = Div(self)
            self.child[0] = Clones(self)
            self.child[2] = Conns(self)
        elif (self.rule == 3):
            self.generate_cells()

    def generate_cells(self):
        # Rule #3 DIV -> CELL CELL CONNS
        self.child[0] = Cell(self)
        self.child[1] = Cell(self)
        self.child[2] = Conns(self)

    def compile(self):
        self.reset()
        self.child[0].ID = self.ID + '0'
        self.child[1].ID = self.ID + '1'
        if (self.rule == 2):
            self.child[1].compile()
            self.child[0].compile()
            self.child[2].compile()
        else:
            self.child[0].compile()
            self.child[1].compile()
            self.child[2].compile()
        self.merge_from_children()

    def update_depth(self):
        self.depth = self.parent.depth + 1
        self.maxDepth = self.depth
        for c in self.child:
            if (c is not None):
                self.maxDepth = max(c.update_depth(), self.maxDepth)
        return self.maxDepth


class Clones(TreeNode):

    def __init__(self, parent):
        super().__init__(parent)
        self.depth += 1

    def generate(self):
        number = random.random()
        if (number < 0.01):
            # rule #0 CLONES -> CLONES CLONE
            self.child[0] = Clone(self)
            self.child[1] = Clones(self)
            self.rule = 0
        else:
            # rule #1 CLONES -> CLONE
            self.child[0] = Clone(self)
            self.rule = 1
        super().generate()

    def compile(self):
        self.reset()
        if (isinstance(self.parent, Div)):
            if (self.parent.rule == 1):
                self.sibling = self.parent.child[0]
            else:
                self.sibling = self.parent.child[1]
        else:
            self.sibling = self.parent.sibling
        if (self.rule == 0):
            self.child[0].ID = self.ID + '0'
            self.child[1].ID = self.ID + '1'
            self.child[0].compile()
            self.child[1].compile()
        else:
            self.child[0].ID = self.ID
            self.child[0].compile()
        self.merge_from_children()

    def update_depth(self):
        self.depth = self.parent.depth + 1
        self.maxDepth = self.depth
        for c in self.child:
            if (c is not None):
                self.maxDepth = max(c.update_depth(), self.maxDepth)
        return self.maxDepth


class Clone(TreeNode):

    def __init__(self, parent):
        super().__init__(parent)

    def generate(self):
        self.child[0] = Conns(self)
        number = random.random()
        self.symmetry = False
        if (number < self.config.symmetry_rate):
            self.permi = np.arange(self.config.input_size)
            chosen = random.choice(self.config.input_cayley)
            for i in range(random.randint(1, chosen[1])):
                self.permi = self.permi[chosen[0]]
            self.permi = self.permi / (self.config.input_size - 1)

            self.permo = np.arange(self.config.output_size)
            chosen = random.choice(self.config.output_cayley)
            for i in range(random.randint(1, chosen[1])):
                self.permo = self.permo[chosen[0]]
            self.permo = self.permo / (self.config.output_size - 1)
            self.symmetry = True

        else:
            self.permi = np.random.rand(self.config.input_size)
            self.permo = np.random.rand(self.config.output_size)
        super().generate()

    def compile(self):
        self.reset()
        self.sibling = self.parent.sibling
        sibDepth = self.sibling.depth
        argPermi = np.argsort(self.permi)
        argPermo = np.argsort(self.permo)
        for neuron, weight in self.sibling.neuronSet.items():
            copy = self.ID + neuron[sibDepth:]
            self.neuronSet[copy] = weight
        for key, value in self.sibling.connSet.items():
            sourceCopy = self.ID + key[0][sibDepth:]
            targetCopy = self.ID + key[1][sibDepth:]
            self.connSet[(sourceCopy, targetCopy)] = value
        for inConn in self.sibling.inSet:
            sourceCopy = argPermi[inConn[0]]
            targetCopy = self.ID + inConn[1][sibDepth:]
            self.inSet.add((sourceCopy, targetCopy, inConn[2]))
        for outConn in self.sibling.outSet:
            sourceCopy = self.ID + outConn[0][sibDepth:]
            targetCopy = argPermo[outConn[1]]
            self.outSet.add((sourceCopy, targetCopy, outConn[2]))
        for c in self.child:
            if (c is not None):
                c.compile()
        self.merge_from_children()


    def mutate(self):
        number = random.random()
        if (number < self.config.conn_relocate_rate):
            pos = random.choice(range(len(self.permi)))
            self.permi[pos] = np.random.rand()
            pos = random.choice(range(len(self.permo)))
            self.permo[pos] = np.random.rand()

    def deepcopy(self, newParent):
        copy = super().deepcopy(newParent)
        copy.permi = self.permi.copy()
        copy.permo = self.permo.copy()
        copy.symmetry = self.symmetry
        return copy


class Conns(TreeNode):

    def __init__(self, parent):
        super().__init__(parent)

    def generate(self):
        number = random.random()
        if (number < 0.5):
            # Rule 0 CONNS -> CONN CONNS
            self.child[0] = Conn(self)
            self.child[1] = Conns(self)
            self.rule = 0
        else:
            # Rule 1 CONNS -> CONN
            self.rule = 1
        super().generate()

    def compile(self):
        self.ID = self.parent.ID
        super().compile()


class Conn(WeightedNode):

    def __init__(self, parent):
        super().__init__(parent)

    def generate(self):
        maxDepth = 2 * self.config.max_depth
        self.sourceTail = bin(random.getrandbits(maxDepth))[2:]
        self.targetTail = bin(random.getrandbits(maxDepth))[2:]
        if (self.config.feedforward):
            # feedforward connection
            self.sourceAddon = '0'
            self.targetAddon = '1'
        else:
            number = random.random()
            if (number < self.config.forward_prob):
                self.sourceAddon = '0'
                self.targetAddon = '1'
            else:
                self.sourceAddon = ''
                self.targetAddon = ''
        self.weight_gen()

    def compile(self):
        self.reset()
        self.ID = self.parent.ID
        d = self.max_depth_from_current() + 1
        source = self.ID + self.sourceAddon + self.sourceTail
        source = source[:d]
        target = self.ID + self.targetAddon + self.targetTail
        target = target[:d]
        self.connSet[(source, target)] = self.weight

    def max_depth_from_current(self):
        p = self.parent
        while(isinstance(p, Conns)):
            p = p.parent
        return p.maxDepth

    def mutate(self):
        number = random.random()
        if (number < self.config.conn_relocate_rate):
            self.generate()
        else:
            self.weight_mut()

    def deepcopy(self, newParent):
        copy = Conn(newParent)
        copy.sourceAddon = self.sourceAddon
        copy.targetAddon = self.targetAddon
        copy.sourceTail = self.sourceTail
        copy.targetTail = self.targetTail
        copy.weight = self.weight
        return copy


class Cell(WeightedNode):

    def __init__(self, parent):
        super().__init__(parent)

    def generate(self):
        self.child[0] = In(self)
        self.child[1] = Out(self)
        self.bias = False
        number = random.random()
        if (number < self.config.bias_rate):
            self.weight_gen()
            self.bias = True
        else:
            self.weight = 0
        self.act = np.random.choice(self.config.avail_acts)
        super().generate()

    def compile(self):
        self.reset()
        self.neuronSet[self.ID] = (self.weight, self.act)
        self.child[0].ID = self.ID
        self.child[1].ID = self.ID
        self.child[0].compile()
        self.child[1].compile()
        self.merge_from_children()

    def mutate(self):
        number = random.random()
        if (number < 0.01):
            self.bias = not self.bias
        if (self.bias):
            self.weight_mut()
        else:
            self.weight = 0
        number = random.random()
        if (number < self.config.acts_mut_rate):
            self.act = np.random.choice(self.config.avail_acts)

    def deepcopy(self, newParent):
        copy = Cell(newParent)
        copy.ID = self.ID
        copy.weight = self.weight
        copy.act = self.act
        copy.bias = self.bias
        for i in range(3):
            if (self.child[i] is not None):
                copy.child[i] = self.child[i].deepcopy(copy)
        return copy



class In(WeightedNode):
    def __init__(self, parent):
        super().__init__(parent)

    def generate(self):
        number = random.random()
        if (number < 0.5):
            # Rule #0 No connection
            self.rule = 0
        else:
            # Rule #1 In -> IO in?
            self.rule = 1
            self.source = random.choice(range(0, self.config.input_size))
        self.weight_gen()

    def compile(self):
        self.reset()
        if (self.rule == 1):
            self.inSet.add((self.source, self.ID, self.weight))

    def mutate(self):
        self.weight_mut()

    def deepcopy(self, newParent):
        copy = In(newParent)
        copy.rule = self.rule
        if (self.rule == 1):
            copy.source = self.source
            copy.weight = self.weight
        return copy


class Out(WeightedNode):
    def __init__(self, parent):
        super().__init__(parent)

    def generate(self):
        number = random.random()
        if (number < 0.5):
            # Rule #0 No connection
            self.rule = 0
        else:
            # Rule #1 Out -> IO out?
            self.rule = 1
            self.target = random.choice(range(0, self.config.output_size))
        self.weight_gen()

    def compile(self):
        self.reset()
        if (self.rule == 1):
            self.outSet.add((self.ID, self.target, self.weight))

    def mutate(self):
        self.weight_mut()

    def deepcopy(self, newParent):
        copy = Out(newParent)
        copy.rule = self.rule
        if (self.rule == 1):
            copy.target = self.target
            copy.weight = self.weight
        return copy
