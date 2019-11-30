# para crear copias de los arboles
from copy import deepcopy
# random
import random


def is_function(f):
    return hasattr(f, "__call__")


def chunks(iterable, n):
    for i in range(0, len(iterable), n):
        yield iterable[i:i + n]


class Node:
    def __init__(self, function):
        assert is_function(function)
        self.operation = function
        self.num_arguments = function.__code__.co_argcount
        self.arguments = []

    def eval(self, dict=None):
        # Se agrega dict como un argumento opcional
        assert len(self.arguments) == self.num_arguments
        return self.operation(*[node.eval(dict) for node in self.arguments])

    def serialize(self):
        l = [self]
        for node in self.arguments:
            l.extend(node.serialize())
        return l

    def copy(self):
        return deepcopy(self)

    def replace(self, otherNode):
        assert isinstance(otherNode, Node)
        self.__class__ = otherNode.__class__
        self.__dict__ = otherNode.__dict__


class BinaryNode(Node):
    num_args = 2

    def __init__(self, function, left, right):
        assert isinstance(left, Node)
        assert isinstance(right, Node)
        super(BinaryNode, self).__init__(function)
        self.arguments.append(left)
        self.arguments.append(right)
        

class AddNode(BinaryNode):
    def __init__(self, left, right):
        def _add(x,y):
            return x + y
        super(AddNode, self).__init__(_add, left, right)

    def __repr__(self):
        return "({} + {})".format(*self.arguments)
        
    
class SubNode(BinaryNode):
    def __init__(self, left, right):
        def _sub(x,y):
            return x - y
        super(SubNode, self).__init__(_sub, left, right)
        
    def __repr__(self):
        return "({} - {})".format(*self.arguments)
    
    
class MaxNode(BinaryNode):
    def __init__(self, left, right):
        def _max(x,y):
            return max(x,y)
        super(MaxNode, self).__init__(_max, left, right)
        
    def __repr__(self):
        return "max({{{}, {}}})".format(*self.arguments)


class MultNode(BinaryNode):
    def __init__(self, left, right):
        def _mult(x,y):
            return x * y
        super(MultNode, self).__init__(_mult, left, right)
        
    def __repr__(self):
        return "({} * {})".format(*self.arguments)


# -------------------------------------------------------------------- #
# -------------------- CREADOS Y MODIFICADOS ------------------------- #
# -------------------------------------------------------------------- #

class DivNode(BinaryNode):
    def __init__(self, left, right):
        def _div(x,y):
            try:
                return x / y
            except ZeroDivisionError:  # Se atrapa la excepción
                return float("inf")  # Se retorna un valor que permite 'castigar' al arbol
        super(DivNode, self).__init__(_div, left, right)

    def __repr__(self):
        return "({} / {})".format(*self.arguments)


class TerminalNode(Node):
    num_args = 0

    def __init__(self, value):
        def _nothind(): pass
        super(TerminalNode, self).__init__(_nothind)
        self.value = value
        
    def __repr__(self):
        return str(self.value)
    
    def eval(self, dict=None):
        # Se agrega dict como un argumento opcional
        # Si value es de tipo str, se considera una variable y por lo tanto se
        # busca su valor en el diccionario dict. De otra forma se retorna ese valor.
        return dict[self.value] if type(self.value) == str else self.value
