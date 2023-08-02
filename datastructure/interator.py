from typing import Callable

class IterQueue(object):
    def __init__(self, l: list) -> None:   
        self.l: list = l

    def pop(self):
        return self.l.pop(0)
    
    def append(self, item):
        self.l.append(item)

    def empty(self):
        return len(self.l) == 0

    def get_not_pop(self):
        return self.l[0]
    
    def get_queue(self):
        return self.l
    
    def pop_queue(self):
        res = self.get_queue()
        self.l = []
        return res

    def set_queue(self, ll):
        self.l = ll

    def lenth(self):
        return len(self.l)
    
def broad_first_iter(o, iter_queue: IterQueue, iter_f: Callable, added_condition: Callable, *args):
    while not iter_queue.empty():
        node = iter_queue.pop()
        yield node
        for subnode in iter_f(node):
            if added_condition(subnode, *args):
                iter_queue.append(subnode)

def improved_brofir_iter(o, iter_queue: IterQueue, iter_f: Callable, added_condition: Callable, *args):
    while not iter_queue.empty():
        node = iter_queue.get_not_pop()
        yield node
        for subnode in iter_f(iter_queue):
            if added_condition(subnode, *args):
                iter_queue.append(subnode)