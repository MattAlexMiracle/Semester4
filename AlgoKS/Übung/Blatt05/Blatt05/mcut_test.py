#!/usr/bin/env python3

import numpy as np

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

    check("Aufgabe 2a: Schnittdimension", 2, "mcut",
          imports = ["numpy", "matplotlib.pyplot", "imgutils"],
          calls = [
              (("cut_dimension", ((1,1), (2,2), (3,3))), 0),
              (("cut_dimension", ((1,1), (2,3), (3,4))), 1),
              (("cut_dimension", ((1,1), (2,3), (4,6))), 2),
              (("cut_dimension", ((1,9), (2,3), (4,6))), 0),
              (("cut_dimension", ((1,9), (2,255), (4,6))), 1),
              (("cut_dimension", ((100,105), (90,99), (95,105))), 2),
          ])

    check("Aufgabe 2b: Median Cut", 3, "mcut",
          imports = ["numpy", "matplotlib.pyplot", "imgutils"],
          calls = [
              (("recursive_median_cut", [], 100), []),
              (("recursive_median_cut", [(0,0,0,0,0)], 100), [(0,0,0,0,0)]),
              (("recursive_median_cut", ([(0,0,0,0,0), (1,1,1,1,0), (2,2,2,2,0), (3,3,3,3,0)]), 1, False),
               [(0,0,0,0,0), (0,0,0,1,0), (2,2,2,2,0), (2,2,2,3,0)]),
              (("recursive_median_cut", ([(0,0,0,0,0), (1,1,1,1,0), (2,2,2,2,0), (3,3,3,3,0)]), 1, ((0,3),(0,3),(0,3))),
               [(0,0,0,0,0), (0,0,0,1,0), (2,2,2,2,0), (2,2,2,3,0)]),
              (("median_cut", np.array([[[5,5,3]]]), 1), np.array([[[5,5,3]]])),
              (("median_cut", np.array([[[5,5,3]]]), 4), np.array([[[5,5,3]]])),
              (("median_cut", np.array([[[0,0,0],[1,1,1],[2,2,2]]]), 1), np.array([[[0,0,0], [2,2,2], [2,2,2]]])),
              (("median_cut", np.array([[[0,0,0],[1,1,1],[2,2,2],[3,3,3]]]), 1), np.array([[[0,0,0], [0,0,0], [2,2,2], [2,2,2]]])),
              (("median_cut", np.array([[[0,0,0],[0,0,0]],[[2,2,2],[2,3,2]]]), 1), np.array([[[0,0,0], [0,0,0]],[[2,2,2], [2,2,2]]])),
              (("median_cut", np.array([[[0,0,0],[0,0,0]],[[2,2,2],[2,5,2]]]), 1), np.array([[[0,0,0], [0,0,0]],[[2,4,2], [2,4,2]]])),
              (("median_cut", np.array([[[0,0,0],[0,0,0]],[[2,2,2],[2,3,2]]]), 2), np.array([[[0,0,0], [0,0,0]],[[2,2,2], [2,3,2]]])),
              (("median_cut", np.array([[[  0,  0,  0],[128,128,128]],
                                        [[255,255,255],[192, 64,221]]]), 2),
               np.array([[[  0,  0,  0],[128,128,128]],
                         [[255,255,255],[192, 64,221]]])),
              (("median_cut", np.array([[[155, 21,163],[ 53,122,224],[ 26,237,155]],
                                        [[173,166, 73],[ 87, 46,129],[106, 69,192]]]), 2),
               np.array([[[130, 45,178],[ 40,180,190],[ 40,180,190]],
                         [[173,166, 73],[ 87, 46,129],[130, 45,178]]])),
          ])

    report()