#!/usr/bin/env python
# coding: utf-8

# In[1]:


import random
import numpy as np
import uuid
from enum import Enum, auto
import anytree
import functools
import collections
from tqdm.autonotebook import tqdm


# # Lib Definitions

# In[2]:


def get_node_id():
    return str(uuid.uuid4())[:8]

class Node(anytree.Node):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, _id=get_node_id(), **kwargs)

    def print(self):
        print(anytree.RenderTree(self).by_attr(lambda n: str(n.name)))


# In[3]:


class DomainSpecificLanguage(object):
    def __init__(self):
        self.symbols = None
        self.start_symbol = None
        self.rules = collections.defaultdict(dict)
        
    def set_symbols(self, symbols):
        if self.symbols is None:
            self.symbols = symbols
        else:
            raise ValueError("Symbols can't be reset")
            
    def set_start_symbol(self, symbol):
        assert isinstance(symbol, self.symbols)
        self.start_symbol = symbol
            
    def add_rule(self, symbol, to, renderer=None):
        assert symbol in self.symbols
        if not isinstance(to, tuple):
            if isinstance(to, str):
                to = (to,)
            else:
                raise ValueError
        if renderer is None:
            renderer = lambda x: x
        self.rules[symbol][to] = renderer
    
    def get_start_node(self):
        return Node(self.symbols.BOOL_EXP)
    
    def get_available_expansions(self, symbol):
        if isinstance(symbol, self.symbols):
            return list(self.rules[symbol].keys())
        else:
            return []
    
    def expand(self, node, expansion):
        assert expansion in self.get_available_expansions(node.name)
        children = []
        if not isinstance(expansion, tuple):
            expansion = (expansion,)
        for token in expansion:
            child = Node(token, parent=node)
            children.append(child)
        return children
    
    def render(self, node):
        children_names = tuple(n.name for n in node.children)
        if node.name in self.rules:
            assert isinstance(node.name, self.symbols)
            renderer = self.rules[node.name][children_names]
            renderer_args = [self.render(child) for child in node.children]
            return renderer(*renderer_args)
        else:
            assert isinstance(node.name, str)
            return node.name
        
    def bnf(self):
        lines = []
        for symbol in self.rules:
            head = f"<{symbol.name}> ::= "
            branches = []
            for branch in self.rules[symbol]:
                branch_tokens = [self._to_bnf_token(item) for item in branch]
                branches.append(' '.join(branch_tokens))
            tail = ' | '.join(branches)
            lines.append(head + tail)
        return '\n'.join(lines)
        
    def _to_bnf_token(self, item):
        if isinstance(item, self.symbols):
            return f"<{item.name}>"
        elif isinstance(item, str):
            return f"\"{item}\""
        else:
            raise ValueError


# In[4]:


def expand_randomly(node):
    available_expansions = dsl.get_available_expansions(node.name)
    if available_expansions:
        expansion = random.choice(available_expansions)
        children = dsl.expand(node, expansion)
        for child in children:
            expand_randomly(child)
    return node


# In[5]:


class Synthesizer(object):
    def __init__(self, dsl):
        self.dsl = dsl
        
    def synthesize(self, task):
        pass


# In[6]:


class RandomSynthesizer(Synthesizer):
    def synthesize(self, task):
        start = self.dsl.get_start_node()
        return self.dsl.render(expand_randomly(start))


# In[7]:


class RandomBestFitSynthesizer(Synthesizer):
    def __init__(self, dsl, executor, num_trials):
        self.dsl = dsl
        self.executor = executor
        self.num_trials = num_trials
        
    def synthesize(self, task):
        curr_best_acc = 0
        curr_best_program = None
        for i in range(self.num_trials):
            start = self.dsl.get_start_node()
            program = self.dsl.render(expand_randomly(start))
            results = []
            for x, y in zip(task['x'], task['y']):
                result = self.executor.run(program, feed={'x': x}) == y 
                results.append(result)
            acc = np.mean(results).item()
            if acc >= curr_best_acc:
                curr_best_acc = acc
                curr_best_program = program
        return curr_best_program


# In[8]:


class Executor(object):
    def run(self, program, feed=None):
        pass

class PythonExecutor(Executor):
    def __init__(self, sketch=None):
        self.sketch = sketch
        
    def run(self, program, feed=None):
        if self.sketch:
            program = self.sketch.format(program)
        
        temp_locals = {'feed': feed}
        exec(program, globals(), temp_locals)
        return temp_locals['result']


# In[9]:


# executor tests
sketch = r"""
def foo():
    x = [0, 1, 2, 3]
    return {0}(x)
    
result = foo()
"""
executor = PythonExecutor(sketch)
print(executor.run("max"))
print(executor.run("min"))
print(executor.run("all"))
print(executor.run("any"))


# # Data Creation 

# In[10]:


def get_random_operation(input_len):
    op_pool = []
    for i in range(input_len):
        for j in range(input_len):
            if i != j:
                and_func = lambda x, i=i, j=j: x[i] and x[j]
                and_func.str = f"(x[{i}] and x[{j}])"
                op_pool.append(and_func)
                or_func = lambda x, i=i, j=j: x[i] or x[j]
                or_func.str = f"(x[{i}] or x[{j}])"
                op_pool.append(or_func)
        not_func = lambda x, i=i: not x[i]
        not_func.str = f"not(x[{i}])"
        op_pool.append(not_func)
    num_ops = random.randint(1, 4)
    ops = [random.choice(op_pool) for _ in range(num_ops)]
    def func(x):
        val = True
        for op in ops:
            val = val and op(x)
        return val
    return func, ops

def get_tasks(input_len=5, num_tasks=100, num_samples_per_task=100):
    data = []
    for _ in tqdm(range(num_tasks)):
        task_true_func, ops = get_random_operation(input_len=input_len)
        task_data = {'x': [], 'y': [], 'true_ops': ops}
        for _ in range(num_samples_per_task):
            x = [random.choice([True, False]) for _ in range(input_len)]
            y = task_true_func(x)
            task_data['x'].append(x)
            task_data['y'].append(y)
        data.append(task_data)
    return data


# # Testing

# ## DSL definition

# In[11]:


class Symbol(Enum):
    BOOL_EXP = auto()
    AND = auto()
    OR = auto()
    NOT = auto()
    BOOL = auto()

dsl = DomainSpecificLanguage()
dsl.set_symbols(Symbol)
dsl.set_start_symbol(Symbol.BOOL_EXP)
dsl.add_rule(
    symbol=Symbol.BOOL_EXP,
    to=(Symbol.BOOL, Symbol.AND, Symbol.BOOL),
    renderer=lambda lhs, and_, rhs: f"({lhs} {and_} {rhs})"
)
dsl.add_rule(
    symbol=Symbol.BOOL_EXP,
    to=(Symbol.BOOL, Symbol.OR, Symbol.BOOL),
    renderer=lambda lhs, or_, rhs: f"({lhs} {or_} {rhs})"
)
dsl.add_rule(
    symbol=Symbol.BOOL_EXP,
    to=(Symbol.NOT, Symbol.BOOL),
    renderer=lambda not_, bool_: f"{not_}({bool_})"
)
dsl.add_rule(symbol=Symbol.AND, to="and")
dsl.add_rule(symbol=Symbol.OR, to="or")
dsl.add_rule(symbol=Symbol.NOT, to="not")
for i in range(5):
    dsl.add_rule(symbol=Symbol.BOOL, to=f"x[{str(i)}]")
    
print(dsl.bnf())


# In[12]:


sketch = r"""
def compute(x):
    return {0}
result = compute(feed["x"])
"""
executor = PythonExecutor(sketch)


# ## test on data

# In[13]:


def evaluate(synthesizer, tasks):
    task_results = []
    for task in tasks:
        split = int(0.8 * len(task['x']))
        train_x, train_y = task['x'][:split], task['y'][:split]
        test_x, test_y = task['x'][split:], task['y'][split:]

        program = synthesizer.synthesize({'x': train_x, 'y': train_y})
        task_result = []
        for x, y in zip(test_x, test_y):
            output = executor.run(program, feed={"x": x})
            task_result.append(output == y)
        task_results.append(task_result)
    return task_results


# In[14]:


tasks = get_tasks(num_tasks=100, num_samples_per_task=100)


# In[15]:


# random baseline
syn = RandomSynthesizer(dsl)
results = evaluate(syn, tasks)
acc = np.array(results).flatten().mean()
print(acc)


# In[16]:


for num_trials in [4, 8, 16, 32]:
    syn = RandomBestFitSynthesizer(dsl, executor=executor, num_trials=num_trials)
    results = evaluate(syn, tasks)
    acc = np.array(results).flatten().mean()
    print(num_trials, acc)


# In[17]:


for _ in range(20):
    num_trials = 64
    task = random.choice(tasks)
    syn = RandomBestFitSynthesizer(dsl, executor=executor, num_trials=num_trials)
    program = syn.synthesize(task)
    print("generated program: ", program)
    print("actual program: ", ' and '.join(op.str for op in task['true_ops']))
    print()

