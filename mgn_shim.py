"""TF2 compatibility shim for DeepMind MeshGraphNets.

WHY THIS EXISTS
---------------
The official meshgraphnets code (deepmind-research/meshgraphnets) is written
against Sonnet 1.x (`snt.AbstractModule`) and pins `dm-sonnet<2`. Sonnet 2
removed `AbstractModule` entirely, so we must use `dm-sonnet==1.36`.

Sonnet 1.36 in turn contains nine *module-level* imports of `tensorflow.contrib`
with no try/except guard, e.g.:

    sonnet/python/modules/base.py:53     from tensorflow.contrib.eager.python import tfe
    sonnet/python/modules/basic.py:34    from tensorflow.contrib import framework
    sonnet/python/modules/residual.py:25 from tensorflow.contrib import framework
    sonnet/python/ops/nest.py:27         from tensorflow.contrib import framework

`tensorflow.contrib` was deleted in TF2, so `import sonnet` dies with
ModuleNotFoundError before any of our code runs.

This module installs a meta-path finder that fabricates `tensorflow.contrib.*`
on demand, mapping each requested symbol onto its surviving `tf.compat.v1` /
`tf.nest` equivalent. Every symbol below was read out of the actual Sonnet 1.36
source, so this is a mapped surface, not a guess.

The finder is lazy: it does not import TensorFlow until something actually tries
to import `tensorflow.contrib`, so plain `python -c "print(1)"` stays fast.

Activated via `site-packages/sitecustomize.py`, which CPython imports at
interpreter startup. The official meshgraphnets .py files stay byte-identical.
"""

import importlib.abc
import importlib.machinery
import sys
import types


_TF1_SEMANTICS_DONE = False


def _tf():
    """Import TensorFlow lazily, force TF1 semantics, return (tf, tf.compat.v1)."""
    global _TF1_SEMANTICS_DONE
    import tensorflow as tf

    tf1 = tf.compat.v1

    if not _TF1_SEMANTICS_DONE:
        _TF1_SEMANTICS_DONE = True
        # The released code was written against TF 1.15, where there was no v2
        # behaviour to opt out of. Under TF 2.x the rollout path in
        # cloth_eval.py / cfd_eval.py relies on TF1-style `tf.while_loop` +
        # `tf.TensorArray` and TF1 tensor shapes; control-flow v2 and
        # TensorShape v2 both change that behaviour. Upstream's own TF2 port
        # (deepmind-research PR #745) calls these two as well.
        #
        # run_model.py already calls `tf.disable_eager_execution()` itself in
        # main(); these two are not in the official source, so we apply them
        # here rather than editing the released files.
        for fn in ("disable_control_flow_v2", "disable_v2_tensorshape"):
            getattr(tf1, fn, lambda: None)()

    return tf, tf1


def _first(*candidates):
    """Return the first non-None candidate.

    Sonnet 1.36 was written against TF 1.15, and a few symbols it reaches for
    moved or were renamed in TF2 (`tf.contrib.eager.defun` -> `tf.function`).
    Resolving defensively means one missing name degrades to None instead of
    blowing up `import sonnet` for every consumer.
    """
    for c in candidates:
        if c is not None:
            return c
    return None


_NEST_MODULE = None


def _nest_attrs():
    """Attrs dict for the cached `tensorflow.contrib.framework.nest` module."""
    global _NEST_MODULE
    if _NEST_MODULE is None:
        _NEST_MODULE = _build_nest_module(_tf()[0])
    return vars(_NEST_MODULE)


def _build_nest_module(tf):
    """Rebuild the TF1 `tf.contrib.framework.nest` API on top of `tf.nest`.

    TF2's `tf.nest` is a trimmed reimplementation. The contrib-era names below
    are gone, and Sonnet 1.36 wraps every one of them at import time in
    `sonnet/python/ops/nest.py`. Only `is_sequence` is ever actually called
    (sonnet/python/modules/basic.py:1398); the rest must merely exist, but they
    are cheap to implement correctly so they are not left as traps.
    """
    nest = types.ModuleType("tensorflow.contrib.framework.nest")
    tfn = tf.nest

    def is_sequence(x):
        # contrib `is_sequence` == TF2 `is_nested`.
        return tfn.is_nested(x)

    def flatten_up_to(shallow_tree, input_tree, check_types=True,
                      expand_composites=False):
        if not tfn.is_nested(shallow_tree):
            return [input_tree]
        if check_types and type(shallow_tree) is not type(input_tree):
            raise ValueError("shallow_tree and input_tree have different types: "
                             "%r vs %r" % (type(shallow_tree), type(input_tree)))
        if isinstance(shallow_tree, dict):
            out = []
            for key in shallow_tree:
                out.extend(flatten_up_to(shallow_tree[key], input_tree[key],
                                         check_types))
            return out
        if isinstance(shallow_tree, (list, tuple)):
            if len(shallow_tree) != len(input_tree):
                raise ValueError("shallow_tree and input_tree have different "
                                 "lengths: %d vs %d"
                                 % (len(shallow_tree), len(input_tree)))
            out = []
            for sub_shallow, sub_input in zip(shallow_tree, input_tree):
                out.extend(flatten_up_to(sub_shallow, sub_input, check_types))
            return out
        return [input_tree]

    def map_structure_up_to(shallow_tree, func, *inputs, **kwargs):
        if not tfn.is_nested(shallow_tree):
            return func(*inputs, **kwargs)
        if isinstance(shallow_tree, dict):
            return {k: map_structure_up_to(shallow_tree[k], func,
                                           *[i[k] for i in inputs], **kwargs)
                    for k in shallow_tree}
        if isinstance(shallow_tree, (list, tuple)):
            return type(shallow_tree)(
                map_structure_up_to(sub, func, *[i[j] for i in inputs], **kwargs)
                for j, sub in enumerate(shallow_tree)
            )
        return func(*inputs, **kwargs)

    def flatten_dict_items(dictionary):
        """{'a': {'b': 1}} -> {('a', 'b'): 1}, as in contrib."""
        result = {}
        for key, value in dictionary.items():
            if isinstance(value, dict):
                for sub_key, sub_value in flatten_dict_items(value).items():
                    result[(key,) + sub_key] = sub_value
            else:
                result[(key,)] = value
        return result

    def assert_shallow_structure(shallow_tree, input_tree, check_types=True,
                                 expand_composites=False):
        flatten_up_to(shallow_tree, input_tree, check_types)

    nest.is_nested = tfn.is_nested
    nest.is_sequence = is_sequence
    nest.assert_same_structure = tfn.assert_same_structure
    nest.flatten = tfn.flatten
    nest.pack_sequence_as = tfn.pack_sequence_as
    nest.map_structure = tfn.map_structure
    nest.flatten_up_to = flatten_up_to
    nest.map_structure_up_to = map_structure_up_to
    nest.flatten_dict_items = flatten_dict_items
    nest.assert_shallow_structure = assert_shallow_structure
    return nest


def _framework_attrs():
    tf, tf1 = _tf()
    graph_keys = getattr(tf1, "GraphKeys", None)
    global _NEST_MODULE
    if _NEST_MODULE is None:
        _NEST_MODULE = _build_nest_module(tf)
    return dict(
        nest=_NEST_MODULE,
        map_structure=tf.nest.map_structure,
        smart_cond=getattr(tf1, "smart_cond", None),
        get_variables=lambda: tf1.get_collection(graph_keys.GLOBAL_VARIABLES),
        get_trainable_variables=lambda: tf1.get_collection(
            graph_keys.TRAINABLE_VARIABLES
        ),
        get_model_variables=lambda: tf1.get_collection(graph_keys.MODEL_VARIABLES),
        get_local_variables=lambda: tf1.get_collection(graph_keys.LOCAL_VARIABLES),
        get_or_create_global_step=tf1.train.get_or_create_global_step,
        get_global_step=lambda: tf1.train.get_global_step(),
        is_tensor=tf.is_tensor,
        is_variable=lambda x: isinstance(x, tf1.Variable),
        with_doc=getattr(tf1, "with_doc", None),
        add_arg_scope=getattr(tf1, "add_arg_scope", None),
        arg_scope=getattr(tf1, "arg_scope", None),
    )


def _tfe_attrs():
    """`tensorflow.contrib.eager.python.tfe` -> used at base.py:395 `tfe.defun`.

    There is no `tf.compat.v1.defun` in TF2; `tf.contrib.eager.defun` became
    `tf.function` (still aliased as `tf.defun`). Only reached on Sonnet's
    defun-wrapped eager path, which this graph-mode reproduction never enters --
    but the attribute must exist for the module to be built at all.
    """
    tf, tf1 = _tf()
    return dict(
        defun=_first(getattr(tf, "defun", None), tf.function),
        enable_eager_execution=_first(
            getattr(tf1, "enable_eager_execution", None),
            getattr(tf, "enable_eager_execution", None),
        ),
        enable_eager_execution_internal=_first(
            getattr(tf1, "enable_eager_execution", None),
            getattr(tf, "enable_eager_execution", None),
        ),
    )


def _rnn_attrs():
    """`tensorflow.contrib.rnn` -> gated_rnn.py:1781 uses LSTMBlockCell."""
    tf, tf1 = _tf()
    cell = _first(getattr(tf1.nn, "rnn_cell", None), getattr(tf, "nn", None))
    return dict(
        LSTMBlockCell=getattr(cell, "LSTMBlockCell", None),
        RNNCell=getattr(cell, "RNNCell", None),
        LSTMCell=getattr(cell, "LSTMCell", None),
        GRUCell=getattr(cell, "GRUCell", None),
    )


# name -> (builder, is_package)
_SPECS = {
    "tensorflow.contrib": (lambda: {}, True),
    "tensorflow.contrib.framework": (_framework_attrs, True),
    # Also registered as a side effect of _framework_attrs; listed so a direct
    # `import tensorflow.contrib.framework.nest` resolves to the same object.
    "tensorflow.contrib.framework.nest": (_nest_attrs, True),
    "tensorflow.contrib.eager": (lambda: {}, True),
    "tensorflow.contrib.eager.python": (lambda: {}, True),
    "tensorflow.contrib.eager.python.tfe": (_tfe_attrs, True),
    "tensorflow.contrib.rnn": (_rnn_attrs, True),
    # Only referenced in Sonnet docstrings / check_regularizers; permissive stub.
    "tensorflow.contrib.layers": (lambda: {"__getattr__": lambda name: None}, True),
}


class _ContribFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Fabricates `tensorflow.contrib.*` modules on import."""

    def find_spec(self, fullname, path=None, target=None):
        if fullname not in _SPECS:
            return None
        return importlib.machinery.ModuleSpec(fullname, self, is_package=True)

    def create_module(self, spec):
        builder, _ = _SPECS[spec.name]
        module = types.ModuleType(spec.name)
        module.__path__ = []  # mark as a package so submodule imports work
        for key, value in builder().items():
            setattr(module, key, value)
        return module

    def exec_module(self, module):
        pass


def _patch_collections_abc():
    """Restore the ABC aliases that Python 3.10 removed from `collections`.

    Sonnet 1.36 (2019) does `isinstance(custom_getter, collections.Mapping)` at
    sonnet/python/modules/base.py:169. Those aliases were deprecated since 3.3
    and deleted in Python 3.10, so every `snt.AbstractModule.__init__` raises
    `AttributeError: module 'collections' has no attribute 'Mapping'`.
    """
    import collections
    import collections.abc

    for name in (
        "Awaitable", "Coroutine", "AsyncIterable", "AsyncIterator",
        "AsyncGenerator", "Hashable", "Iterable", "Iterator", "Generator",
        "Reversible", "Sized", "Container", "Callable", "Collection",
        "Set", "MutableSet", "Mapping", "MutableMapping", "MappingView",
        "KeysView", "ItemsView", "ValuesView", "Sequence", "MutableSequence",
        "ByteString",
    ):
        if not hasattr(collections, name):
            setattr(collections, name, getattr(collections.abc, name))


def install():
    """Idempotently install the finder at the front of sys.meta_path."""
    _patch_collections_abc()
    if not any(isinstance(f, _ContribFinder) for f in sys.meta_path):
        sys.meta_path.insert(0, _ContribFinder())


install()
