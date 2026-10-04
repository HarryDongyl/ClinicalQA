"""Linear-attention op recorder (D-106): reports the ops a Gated DeltaNet layer actually holds."""

from clinqa.modeling import _linear_attention_ops


def fla_chunk():  # stands in for an fla kernel
    return None


fla_chunk.__module__ = "fla.ops.gated_delta_rule"


class _Layer:
    def __init__(self, conv):
        self.chunk_gated_delta_rule = fla_chunk
        self.recurrent_gated_delta_rule = fla_chunk
        self.causal_conv1d_fn = conv
        self.causal_conv1d_update = None


class _Model:
    def __init__(self, *mods):
        self.mods = mods

    def modules(self):
        return iter(self.mods)


def test_reports_the_bound_ops_of_the_first_linear_attention_layer():
    ops = _linear_attention_ops(_Model(object(), _Layer(conv=None)))
    assert ops == {"chunk_gated_delta_rule": "fla.ops.gated_delta_rule",
                   "recurrent_gated_delta_rule": "fla.ops.gated_delta_rule",
                   "causal_conv1d_fn": None, "causal_conv1d_update": None}


def test_none_without_linear_attention():
    assert _linear_attention_ops(_Model(object(), object())) is None
