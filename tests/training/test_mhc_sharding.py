from zrt.training.ir.builders import build_graph
from zrt.training.search.training_search_util import _load_model_spec
from zrt.training.spec.strategy import Strategy


def _mhc_ops(graph):
    return [
        op for op in graph.ops
        if op.kind in {"mhc_pre", "mhc_post", "mhc_head", "hc_expand"}
    ]


def test_mhc_sequence_metadata_is_sharded_by_tp_and_cp():
    model = _load_model_spec("deepseek_v4_pro")
    model.seq_len = 4096
    strategy = Strategy(tp=4, cp=2, pp=1, ep=1, dp=1)

    graph = build_graph(model, strategy)
    expected_seq = model.seq_len // (strategy.tp * strategy.cp)

    ops = _mhc_ops(graph)
    assert ops, "Expected DeepSeek-V4 HC ops in the graph"
    for op in ops:
        assert op.meta["s"] == expected_seq, op.name


def test_mhc_sequence_tensor_shapes_are_sharded_by_tp_and_cp():
    model = _load_model_spec("deepseek_v4_pro")
    model.seq_len = 4096
    strategy = Strategy(tp=4, cp=2, pp=1, ep=1, dp=1)

    graph = build_graph(model, strategy)
    expected_seq = model.seq_len // (strategy.tp * strategy.cp)

    for op in _mhc_ops(graph):
        for tensor in op.inputs + op.outputs:
            if tensor.shape_logical and tensor.shape_logical[0] == model.seq_len:
                assert tensor.shape_local[0] == expected_seq, (
                    op.name,
                    tensor.name,
                    tensor.shape_local,
                )


def test_rmsnorm_hidden_dimension_is_sharded_by_tp():
    model = _load_model_spec("deepseek_v4_pro")
    strategy = Strategy(tp=4, cp=1, pp=1, ep=1, dp=1)

    graph = build_graph(model, strategy)
    expected_hidden = model.hidden // strategy.tp
    rmsnorm_ops = [op for op in graph.ops if op.kind == "rmsnorm" and op.layer_id >= 0]

    assert rmsnorm_ops, "Expected DeepSeek-V4 RMSNorm ops in the graph"
    for op in rmsnorm_ops:
        for tensor in op.inputs + op.outputs:
            if tensor.shape_logical and tensor.shape_logical[-1] == model.hidden:
                assert tensor.shape_local[-1] == expected_hidden, (
                    op.name,
                    tensor.name,
                    tensor.shape_local,
                )
