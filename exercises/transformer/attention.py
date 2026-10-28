"""Learner workspace: causal single-head attention, without dropout or biases.

Contract: x is [batch, tokens, width]; each weight is [width, width].
Return output [batch, tokens, width] and weights [batch, tokens, tokens].

TODO(Step 1.1): Compute the query, key and value projections of x.
Hint: each projection has its own matrix; transform the last axis.
Done when: all three shapes match x and changing one matrix affects its projection.

TODO(Step 1.2): Compute scaled pairwise query/key scores.
Hint: compare positions within each batch; scale by the square root of width.
Done when: scores have shape [batch, tokens, tokens].

TODO(Step 1.3): Apply a causal mask and normalize scores into attention weights.
Hint: each row is a query position; future key positions are unavailable.
Done when: future weights are zero and each row sums to approximately one.

TODO(Step 1.4): Combine values using the attention weights; return both results.
Done when: output has the contracted shape and finite values.

TODO(Step 1.5): Write meaningful checks in check_attention below.
Cover a one-token case, a hand-checkable two-token case, batch isolation,
and changing future inputs without changing earlier outputs.
Done when: checks pass; record errors and explain any repaired failure.

Begin with Step 1.1 only. Ask for explanations and hints as needed.
"""


def single_head_attention(x, w_q, w_k, w_v):
    raise NotImplementedError("Start with Step 1.1: Q/K/V projections")


def check_attention():
    raise NotImplementedError("Step 1.5: learner-written correctness checks")


if __name__ == "__main__":
    import torch

    torch.manual_seed(42)
    x = torch.randn(1, 4, 16)
    w_q = torch.randn(16, 16)
    w_k = torch.randn(16, 16)
    w_v = torch.randn(16, 16)
    print("Input shape:", x.shape)
    output, weights = single_head_attention(x, w_q, w_k, w_v)
    print("Output shape:", output.shape)
    print("Attention shape:", weights.shape)
