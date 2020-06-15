#!/usr/bin/env python3

import os
import sys
import importlib
import textwrap
import numpy as np
import math as m
import multiprocessing as mp
import ast
import pprint
import importlib.util
import time
import math
import numbers


max_points      = 0
obtained_points = 0
default_permitted_modules = ["matplotlib",
                             "matplotlib.pyplot",
                             "matplotlib.widgets",
                             "functools",
                             "mpl_toolkits.mplot3d"]
timeout = 60
numeric_accuracy = 1e-6


def info(string):
    print(string)


def format_beautifully(*blocks):
    # step #1: convert short atomics to wrap statements
    preprocessed_blocks = []
    for (kind, object) in blocks:
        if   kind == 'wrap':
            preprocessed_blocks += [(kind, str(object))]
        elif (kind == 'atomic') or (kind == 'args'):
            string = pprint.pformat(object, width=79)
            if kind == 'args':
                string = "(" + string[1:-1] + ")"
            if '\n' in string:
                preprocessed_blocks += [('atomic', string)]
            else:
                preprocessed_blocks += [('wrap', string)]

    # step #2: merge consecutive wrap statements
    merged_blocks = []
    wrap_strings = []
    for (kind, string) in preprocessed_blocks:
        if   kind == 'wrap':
            wrap_strings += [string]
        elif kind == 'atomic':
            merged_blocks += [('wrap', "".join(wrap_strings))]
            wrap_strings = []
            merged_blocks += [(kind, string)]
    merged_blocks += [('wrap', "".join(wrap_strings))]

    # step #3: combine the text blocks
    #          in particular merge atomics followed by wrap text
    lines = []
    for (kind, string) in merged_blocks:
        if kind == 'wrap':
            if not lines:
                lines += textwrap.fill(string, width=79).splitlines()
            else:
                indent = len(lines[-1]) if lines else 0
                new_lines = textwrap.fill("#" * indent + string, width=79).splitlines()
                lines[-1] = new_lines[0].replace("#" * indent, lines[-1])
                lines += new_lines[1:]
        elif kind == 'atomic':
            lines += string.splitlines()

    return "\n".join(lines)


def type_equal(a,b):
    # surprise #1: numpy.int64 and the like are not of type integer!
    # surprise #2: numpy.float64 and Python float are disjoint!
    def denumpify(x):
        return np.asscalar(x) if isinstance(x, np.generic) else x
    return type(denumpify(a)) == type(denumpify(b))


def equal(a, b):
    if not type_equal(a,b):
        return False
    if isinstance(a, float):
        return abs(a - b) < numeric_accuracy
    if isinstance(a, np.poly1d):
        return equal(a.c,b.c)
    if isinstance(a, np.ndarray):
        if a.shape != b.shape: return False
        if a.size == 0: return True
        return np.amax(abs(a.astype(np.float64) - b.astype(np.float64))) < numeric_accuracy
    if isinstance(a, tuple) or isinstance(a, list):
        if len(a) != len(b): return False
        for (a, b) in zip(a,b):
            if not equal(a,b): return False
        return True
    return a == b


class CheckFailure(Exception):
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return format_beautifully(('wrap', self.message))


class Timeout(CheckFailure):
    def __init__(self, call):
        self.call = call

    def __str__(self):
        (fname, *args) = self.call
        blocks = [('wrap', "Der Aufruf {}".format(fname)),
                  ('args', args),
                  ('wrap', " terminiert nicht, oder ist SEHR ineffizient.")]
        return format_beautifully(*blocks)


class WrongResult(CheckFailure):
    def __init__(self, call, expected_result, obtained_result):
        self.call = call
        self.expected_result = expected_result
        self.obtained_result = obtained_result

    def __str__(self):
        (fname, *args) = self.call
        blocks = [('wrap', "Der Aufruf {}".format(fname)),
                  ('args', args)]
        if isinstance(self.obtained_result, Exception):
            blocks += ([('wrap', " liefert den Fehler: "),
                        ('atomic', self.obtained_result)])
        elif isinstance(self.obtained_result, type(None)):
            blocks += ([('wrap', " hat keinen Rückgabewert, sollte aber "),
                        ('atomic', self.expected_result),
                        ('wrap', " zurückgeben.")])
        elif not type_equal(self.obtained_result, self.expected_result):
            blocks += ([('wrap', " liefert ein Ergebnis vom Typ "),
                        ('wrap', "'" + type(self.obtained_result).__name__ + "'"),
                        ('wrap', ", erwartet wird aber ein Ergebnis vom Typ "),
                        ('wrap', "'" + type(self.expected_result).__name__ + "'.")])
        else:
            blocks += ([('wrap', " liefert das Ergebnis "),
                        ('atomic', self.obtained_result),
                        ('wrap', ", richtig wäre aber "),
                        ('atomic', self.expected_result),
                        ('wrap', ".")])
        return format_beautifully(*blocks)


class AstInspector(ast.NodeVisitor):
    def __init__(self, file="<unknown>", imports=[]):
        self.file = file
        self.permitted_modules = set(default_permitted_modules + imports)
        self.errors = []


    def check_module(self, module):
        if module in self.permitted_modules: return
        self.errors.append(CheckFailure("Das Modul {} darf für diese "
                                       "Aufgabe nicht verwendet werden."
                                       .format(module)))

    def visit_Import(self, node):
        for m in [n.name for n in node.names]:
            self.check_module(m)


    def visit_ImportFrom(self, node):
        self.check_module(node.module)


def subprocess_eval(queue, proc_id, expr, module):
    try:
        m = importlib.import_module(module)
        (f, *args) = map(lambda x: eval(x, vars(m)) if isinstance(x, str) else x, expr)
        result = f(*args)
    except BaseException as e:
        result = e
    queue.put((proc_id, result))


def check_calls(forms, module_name):
    calls, desired_results = zip(*forms) if forms else ([], [])
    queue = mp.Queue()
    processes = []
    process_ids = set(range(len(calls)))
    for pid in process_ids:
        p = mp.Process(target=subprocess_eval,
                       args=(queue, pid, calls[pid], module_name))
        p.start()
        processes.append(p)

    queue_contents = []
    try:
        for _ in process_ids:
            queue_contents.append(queue.get(True, timeout))
    except:
        pass
    finally:
        for p in processes: p.terminate()

    for pid in process_ids:
        match = [result for (id, result) in queue_contents if id == pid]
        if not match:
            yield Timeout(calls[pid])
        elif not equal(desired_results[pid], match[0]):
            yield WrongResult(calls[pid], desired_results[pid], match[0])


def check_module(module_name, imports=[], calls=[], nuke=[]):
    try:
        spec = importlib.util.find_spec(module_name)
        if not spec:
            yield CheckFailure("Die Datei {}.py konnte nicht gefunden werden. "
                              "Sie sollte im selben Ordner liegen wie "
                              "dieses Programm.".format(module_name))
            return

        with open(spec.origin, 'r', encoding='utf-8') as f:
            source = f.read()
        st = ast.parse(source, spec.origin)
        inspector = AstInspector(spec.origin, imports)
        inspector.visit(st)
        errors = inspector.errors or check_calls(calls, module_name)
        for error in errors: yield error
    except OSError:
        yield CheckFailure("Die Datei {} konnte nicht geöffnet werden."
                          .format(spec.origin))
    except SyntaxError as e:
        yield CheckFailure("Die Datei {} enthält einen Syntaxfehler "
                          "in Zeile {:d}."
                          .format(spec.origin, e.lineno))
    except CheckFailure as e:
        yield e
    except Exception as e:
        yield CheckFailure("Beim Laden des Moduls {} ist ein Fehler "
                          "vom Typ '{}' aufgetreten."
                          .format(module_name, str(e)))


def check(description, points, module, **kwargs):
    global max_points
    global obtained_points

    pstr = ("1 Punkt" if points == 1 else "{:d} Punkte".format(points))
    desc = description + " (" + pstr + ")"
    max_points += points
    info("=" * (len(desc) + 2))
    info(" " + desc)
    info("=" * (len(desc) + 2))

    try:
        stdout, stderr = sys.stdout, sys.stderr
        try:
            with open(os.devnull, 'w') as f:
                sys.stdout, sys.stderr = f, f
                errors = list(check_module(module, **kwargs))
        finally:
            sys.stdout, sys.stderr = stdout, stderr
    except BaseException as e:
        info("Eine unbehandelte Ausnahme ist aufgetreten: '{}'"
             .format(str(e)))
    else:
        if errors:
            for e in errors: info(str(e))
            info("Diese Teilaufgabe enthält Fehler!")
        elif not kwargs["calls"]:
            max_points -= points
            info("Diese Teilaufgabe hat keine öffentlichen Testcases.")
        else:
            obtained_points += points
            info("Super! Alles scheint zu funktionieren.")
    info("")


def import_modules(modules):
    for module in modules:
        importlib.import_module(module)


def compute_process_spawn_time():
    start = time.time()
    p = mp.Process(target=import_modules, args=(default_permitted_modules,))
    p.start()
    p.join()
    end = time.time()
    return math.ceil(end - start)


def report():
    print("Insgesamt: {:d} von {:d} Punkten.".format(obtained_points, max_points))


if __name__ == "__main__":
    mp.freeze_support()
    timeout = 2 * compute_process_spawn_time() + 9


    ###############################
    ## Nun die eigentlichen Checks
    ###############################
    check("Aufgabe 1a: Zeilenvertauschung", 1, "lu",
          imports = ["numpy"],
          calls = [
              (("swap_rows", np.array([[42.]]), 0, 0), np.array([[42.]])),
              (("swap_rows", np.array([[1., 1.], [2., 2.]]), 0, 0),
               np.array([[1., 1.], [2., 2.]])),
              (("swap_rows", np.array([[1., 1.], [2., 2.]]), 0, 1),
               np.array([[2., 2.], [1., 1.]])),
              (("swap_rows", np.array([[ 1., -1.,  1., -1.,  5.],
                                       [-1.,  1., -1.,  4., -1.],
                                       [ 1., -1.,  3., -1.,  1.],
                                       [-1.,  2., -1.,  1., -1.],
                                       [ 1., -1.,  1., -1.,  1.]]), 2, 3),
               np.array([[ 1., -1.,  1., -1.,  5.],
                         [-1.,  1., -1.,  4., -1.],
                         [-1.,  2., -1.,  1., -1.],
                         [ 1., -1.,  3., -1.,  1.],
                         [ 1., -1.,  1., -1.,  1.]])),
              (("swap_rows", np.array([[1., 1., 1.], [2., 2., 2.], [3., 3., 3.]]), 0, 2),
               np.array([[3., 3., 3.], [2., 2., 2.], [1., 1., 1.]])),
          ])

    check("Aufgabe 1b: Zeilen subtrahieren", 1, "lu",
          imports = ["numpy"],
          calls = [
              (("subtract_scaled", np.array([[1.], [1.]]), 1, 0, 1.0),
               np.array([[1.], [0.]])),
              (("subtract_scaled", np.array([[1.], [1.]]), 0, 1, 1.0),
               np.array([[0.], [1.]])),
              (("subtract_scaled", np.array([[1., 1.], [2., 2.]]), 1, 0, 0.0),
               np.array([[1., 1.], [2., 2.]])),
              (("subtract_scaled", np.array([[1., 1.], [2., 2.]]), 1, 0, 2.0),
               np.array([[1., 1.], [0., 0.]])),
              (("subtract_scaled", np.array([[1., 1.], [2., 2.]]), 1, 1, 1.),
               np.array([[1., 1.], [0., 0.]])),
              (("subtract_scaled", np.array([[1., 1.], [2., 2.], [3., 3.]]), 2, 1, 1.),
               np.array([[1., 1.], [2., 2.], [1., 1.]])),
              (("subtract_scaled", np.array([[ 1., -1.,  1., -1.,  5.],
                                             [-1.,  1., -1.,  4., -1.],
                                             [ 1., -1.,  3., -1.,  1.],
                                             [-1.,  2., -1.,  1., -1.],
                                             [ 1., -1.,  1., -1.,  1.]]), 0, 4, 2.),
               np.array([[-1.,  1., -1.,  1.,  3.],
                         [-1.,  1., -1.,  4., -1.],
                         [ 1., -1.,  3., -1.,  1.],
                         [-1.,  2., -1.,  1., -1.],
                         [ 1., -1.,  1., -1.,  1.]])),
     ])

    check("Aufgabe 1c: Pivot Element", 1, "lu",
          imports = ["numpy"],
          calls = [
              (("pivot_index", np.array([[42.]]), 0), 0),
              (("pivot_index", np.array([[1., 2.], [3., 4]]), 0), 1),
              (("pivot_index", np.array([[1., 2.], [3., 4]]), 1), 1),
              (("pivot_index", np.array([[3., 4.], [1., 2]]), 0), 0),
              (("pivot_index", np.array([[3., 9., 9.], [2., 2., 9.], [1., 1., 1.]]), 0), 0),
              (("pivot_index", np.array([[3., 9., 9.], [2., 2., 9.], [1., 1., 1.]]), 1), 1),
              (("pivot_index", np.array([[3., 9., 9.], [2., 2., 9.], [1., 1., 1.]]), 2), 2),
              (("pivot_index", np.array([[-3., 9., 9.], [-2., 2., 9.], [-1., 1., 1.]]), 0), 0),
              (("pivot_index", np.array([[ 1., -1.,  1., -1.,  5.],
                                         [-1.,  1., -1.,  4., -1.],
                                         [ 1., -1.,  3., -1.,  1.],
                                         [-1.,  2., -1.,  1., -1.],
                                         [ 1., -1.,  1., -1.,  1.]]), 1), 3),
          ])

    check("Aufgabe 1d: LR-Zerlegung", 2, "lu",
          imports = ["numpy"],
          calls = [
              (("lu_decompose", np.array([[42.]])),
               (np.array([[1.]]), np.array([[1.]]), np.array([[42.]]))),
              (("lu_decompose", np.array([[1., 1.], [1., 2.]])),
               (np.array([[1., 0.], [0.,1.]]),
                np.array([[1., 0.], [1., 1.]]),
                np.array([[1., 1.], [0., 1.]]))),
              (("lu_decompose", np.array([[1., 3., 5.], [2., 4., 7.], [1.,1.,0.]])),
               (np.array([[0., 1., 0.], [1., 0., 0.], [0., 0., 1.]]),
                np.array([[1., 0., 0.], [0.5, 1., 0.], [0.5, -1., 1.]]),
                np.array([[2., 4., 7.], [0., 1., 1.5], [0.0, 0., -2.]]))),
              (("lu_decompose", np.array([[2., 3., 5.], [6., 10., 17.], [8., 14., 28.]])),
               (np.array([[0., 0., 1.], [0., 1., 0.], [1., 0., 0.]]),
                np.array([[1., 0., 0.], [0.75, 1., 0.], [0.25, 1., 1.]]),
                np.array([[8., 14., 28.], [0., -0.5, -4], [0., 0., 2.]]))),
              (("lu_decompose", np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 0.]])),
               (np.array([[0., 1., 0.], [0., 0., 1.], [1., 0., 0.]]),
                np.array([[1., 0., 0.], [0.14285714, 1., 0.], [0.57142857, 0.5, 1.]]),
                np.array([[7., 8., 0.], [0., 0.85714286, 3.], [0., 0., 4.5]]))),

              (("lu_decompose", np.array([[ 1., -1.,  1., -1.,  5.],
                                          [-1.,  1., -1.,  4., -1.],
                                          [ 1., -1.,  3., -1.,  1.],
                                          [-1.,  2., -1.,  1., -1.],
                                          [ 1., -1.,  1., -1.,  1.]])),
               (np.array([[ 1.,  0.,  0.,  0.,  0.],
                          [ 0.,  0.,  0.,  1.,  0.],
                          [ 0.,  0.,  1.,  0.,  0.],
                          [ 0.,  1.,  0.,  0.,  0.],
                          [ 0.,  0.,  0.,  0.,  1.]]),
                np.array([[ 1.,  0.,  0.,  0.,  0.],
                          [-1.,  1.,  0.,  0.,  0.],
                          [ 1.,  0.,  1.,  0.,  0.],
                          [-1.,  0.,  0.,  1.,  0.],
                          [ 1.,  0.,  0.,  0.,  1.]]),
                np.array([[ 1., -1.,  1., -1.,  5.],
                          [ 0.,  1.,  0.,  0.,  4.],
                          [ 0.,  0.,  2.,  0., -4.],
                          [ 0.,  0.,  0.,  3.,  4.],
                          [ 0.,  0.,  0.,  0., -4.]]))),
              (("lu_decompose",
                np.array([[-0.65556169,  0.34331728,  0.98314055, -0.4550035 ,  0.90004438],
                          [ 0.95241326,  0.84595768,  0.47040399, -0.42291233, -0.96781142],
                          [ 0.53676727, -0.40189718,  0.74985821, -0.61704069, -0.83990612],
                          [-0.0985994 , -0.61068867, -0.67655703, -0.43533887, -0.16422088],
                          [-0.32317589,  0.48831627, -0.58981647,  0.36695473,  0.93373367]])),
               (np.array([[ 0.,  1.,  0.,  0.,  0.],
                          [ 1.,  0.,  0.,  0.,  0.],
                          [ 0.,  0.,  1.,  0.,  0.],
                          [ 0.,  0.,  0.,  1.,  0.],
                          [ 0.,  0.,  0.,  0.,  1.]]),
                np.array([[ 1.        ,  0.        ,  0.        ,  0.        ,  0.        ],
                          [-0.68831642,  1.        ,  0.        ,  0.        ,  0.        ],
                          [ 0.56358651, -0.94929113,  1.        ,  0.        ,  0.        ],
                          [-0.10352586, -0.56515558,  0.06419338,  1.        ,  0.        ],
                          [-0.33932317,  0.83769025, -0.88385204,  0.13508831,  1.        ]]),
                np.array([[  9.52413260e-01,   8.45957680e-01,   4.70403990e-01, -4.22912330e-01,  -9.67811420e-01],
                          [  0.00000000e+00,   9.25603846e-01,   1.30692734e+00, -7.46101003e-01,   2.33883884e-01],
                          [  0.00000000e+00,   0.00000000e+00,   1.72539940e+00, -1.08696007e+00,  -7.24367586e-02],
                          [  0.00000000e+00,   0.00000000e+00,  -1.38777878e-17, -8.31008732e-01,  -1.27583646e-01],
                          [  0.00000000e+00,   0.00000000e+00,   1.87472694e-18, 0.00000000e+00,   3.62622262e-01]]))),
          ])

    check("Aufgabe 1e: Vorwärtseinsetzen", 1, "lu",
          imports = ["numpy"],
          calls = [
              (("forward_substitute", np.array([[1.]]), np.array([[0.]])),
               np.array([[0.]])),
              (("forward_substitute", np.array([[42.]]), np.array([[42.]])),
               np.array([[1.]])),
              (("forward_substitute", np.array([[1., 0.], [2., 1.]]), np.array([[1.], [1.]])),
               np.array([[1.], [-1.]])),
              (("forward_substitute",
                np.array([[ 1.,  0.,  0.,  0.,  0.],
                          [-1.,  1.,  0.,  0.,  0.],
                          [ 1.,  0.,  1.,  0.,  0.],
                          [-1.,  0.,  0.,  1.,  0.],
                          [ 1.,  0.,  0.,  0.,  1.]]),
                np.array([[3.], [1.], [4.], [1.], [5]])),
               np.array([[3.], [4.], [1.], [4.], [2]])),
              (("forward_substitute",
                np.array([[ 1.        ,  0.        ,  0.        ,  0.        ,  0.        ],
                          [-0.68831642,  1.        ,  0.        ,  0.        ,  0.        ],
                          [ 0.56358651, -0.94929113,  1.        ,  0.        ,  0.        ],
                          [-0.10352586, -0.56515558,  0.06419338,  1.        ,  0.        ],
                          [-0.33932317,  0.83769025, -0.88385204,  0.13508831,  1.        ]]),
               np.array([[1.], [1.], [1.], [1.], [1.]])),
               np.array([[ 1.        ],
                         [ 1.68831642],
                         [ 2.03911729],
                         [ 1.92678947],
                         [ 1.46702821]])),
          ])

    check("Aufgabe 1f: Rückwärtseinsetzen", 1, "lu",
          imports = ["numpy"],
          calls = [
              (("backward_substitute", np.array([[1.]]), np.array([[0.]])),
               np.array([[0.]])),
              (("backward_substitute", np.array([[42.]]), np.array([[42.]])),
               np.array([[1.]])),
              (("backward_substitute", np.array([[2., 1.], [0., 1.]]), np.array([[1.], [1.]])),
               np.array([[0.], [1.]])),
              (("backward_substitute", np.array([[2., 1.], [0., 2.]]), np.array([[1.], [1.]])),
               np.array([[0.25], [0.5]])),
              (("backward_substitute", np.array([[ 1., -1.,  1., -1.,  5.],
                                                 [ 0.,  1.,  0.,  0.,  4.],
                                                 [ 0.,  0.,  2.,  0., -4.],
                                                 [ 0.,  0.,  0.,  3.,  4.],
                                                 [ 0.,  0.,  0.,  0., -4.]]),
                np.array([[3.], [4.], [1.], [4.], [2]])),
               np.array([[14.], [6.], [-0.5], [2.], [-0.5]])),
              (("backward_substitute",
               np.array([[  9.52413260e-01,   8.45957680e-01,   4.70403990e-01, -4.22912330e-01,  -9.67811420e-01],
                         [  0.00000000e+00,   9.25603846e-01,   1.30692734e+00, -7.46101003e-01,   2.33883884e-01],
                         [  0.00000000e+00,   0.00000000e+00,   1.72539940e+00, -1.08696007e+00,  -7.24367586e-02],
                         [  0.00000000e+00,   0.00000000e+00,  -1.38777878e-17, -8.31008732e-01,  -1.27583646e-01],
                         [  0.00000000e+00,   0.00000000e+00,   1.87472694e-18, 0.00000000e+00,   3.62622262e-01]]),
               np.array([[1.], [1.], [1.], [1.], [1.]])),
               np.array([[ 3.70344629],
                         [-0.46252673],
                         [-0.32945686],
                         [-1.62674129],
                         [ 2.75769059]])),
          ])

    check("Aufgabe 1g: Gleichungssysteme lösen", 2, "lu",
          imports = ["numpy"],
          calls = [
              (("linsolve", np.array([[1.]]), np.array([[0.]])),
               (np.array([[0.]]),)),
              (("linsolve", np.array([[42.]]), np.array([[42.]])),
               (np.array([[1.]]),)),
              (("linsolve", np.array([[1.]]),
                np.array([[1.]]), np.array([[2.]]), np.array([[3.]])),
               (np.array([[1.]]), np.array([[2.]]), np.array([[3.]]))),
              (("linsolve", np.array([[1., 1., 1.,  1.],
                                      [1., 2., 3.,  4.],
                                      [1., 3., 6.,  10.],
                                      [1., 4., 10., 20.]]),
                np.array([[0.], [0.], [0.], [0.]]),
                np.array([[3.], [1.], [4.], [1.]])),
               (np.array([[0.], [0.], [0.], [0.]]),
                np.array([[21.], [-45.], [38.], [-11.]]))),
          ])

    report()