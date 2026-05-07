"""SparseAttnSharedKV annotation pass for Ascend NPU.

After FusionPass labels V4 Attention nodes as "npu_sas", this pass reads
compress_ratios from graph.metadata and annotates each node with:
  annotations["attn_type"]      : "HCA" | "CSA" | "SWA"
  annotations["compress_ratio"] : int
"""
import re
from python.zrt.transform.base import GraphPass

_LAYER_RE = re.compile(r"layers\.(\d+)\.")


def _attn_type(compress_ratio: int) -> str:
    if compress_ratio == 0:
        return "SWA"
    if compress_ratio == 4:
        return "CSA"
    return "HCA"


class SparseAttnSharedKVPass(GraphPass):
    name = "sas_annotation"

    def run(self, graph, ctx):
        compress_ratios = graph.metadata.get("compress_ratios")
        if not compress_ratios:
            return graph
        g = graph.clone()
        for node in g.topo_sort():
            if node.op_type != "npu_sas":
                continue
            m = _LAYER_RE.search(node.scope or "")
            if m is None:
                continue
            layer_idx = int(m.group(1))
            cr = compress_ratios[layer_idx] if layer_idx < len(compress_ratios) else 0
            node.annotations["attn_type"] = _attn_type(cr)
            node.annotations["compress_ratio"] = cr
        return g
