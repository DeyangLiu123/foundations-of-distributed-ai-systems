/* Canonical card order for the global L12 stream page. */
(function attachL12StreamRegistry(root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  root.L12StreamRegistry = api;
})(typeof globalThis !== "undefined" ? globalThis : this, function buildRegistry() {
  "use strict";

  const item = (source, key, label, kind = "activation") => ({ source, key, label, kind });

  const forward = [
    {
      id: "input", title: "Input / Embedding", subtitle: "token ids → residual stream",
      items: [
        item("a", "token_ids", "token ids", "input"),
        item("a", "target_ids", "next-token targets", "input"),
        item("p", "embedding_table", "embedding table E", "parameter"),
        item("a", "embedding", "embedding X"),
        item("p", "rms1_gamma", "RMSNorm 1 γ", "parameter"),
        item("a", "rms1", "RMSNorm 1 output"),
      ],
    },
    {
      id: "qkv", title: "Q / K / V + RoPE", subtitle: "projection → position → GQA",
      items: [
        item("p", "W_q", "Wq", "parameter"),
        item("p", "W_k", "Wk", "parameter"),
        item("p", "W_v", "Wv", "parameter"),
        item("a", "q_linear", "Q projection"),
        item("a", "k_linear", "K projection"),
        item("a", "v_linear", "V projection"),
        item("a", "q_heads", "Q heads"),
        item("a", "k_heads", "K heads"),
        item("a", "v_heads", "V heads"),
        item("a", "q_rope", "RoPE(Q)"),
        item("a", "k_rope", "RoPE(K)"),
        item("a", "k_repeated", "GQA repeated K"),
        item("a", "v_repeated", "GQA repeated V"),
      ],
    },
    {
      id: "attention", title: "Causal Attention", subtitle: "QKᵀ → softmax → AV",
      items: [
        item("a", "causal_scores", "masked scores"),
        item("a", "attention_weights", "attention weights"),
        item("a", "context_heads", "context per head"),
        item("a", "context", "concatenated context"),
      ],
    },
    {
      id: "residual1", title: "Attention Residual", subtitle: "Wo → add x₀",
      items: [
        item("p", "W_o", "Wo", "parameter"),
        item("a", "attn_out", "attention output"),
        item("a", "residual1", "residual stream x₁", "residual"),
        item("p", "rms2_gamma", "RMSNorm 2 γ", "parameter"),
        item("a", "rms2", "RMSNorm 2 output"),
      ],
    },
    {
      id: "ffn", title: "SwiGLU FFN", subtitle: "gate / up → SiLU ⊙ → down",
      items: [
        item("p", "W_gate", "Wgate", "parameter"),
        item("p", "W_up", "Wup", "parameter"),
        item("a", "gate", "gate projection"),
        item("a", "up", "up projection"),
        item("a", "silu_gate", "SiLU(gate)"),
        item("a", "gated", "gated hidden"),
        item("p", "W_down", "Wdown", "parameter"),
        item("a", "ffn_out", "FFN output"),
        item("a", "residual2", "residual stream x₂", "residual"),
      ],
    },
    {
      id: "output", title: "Final Norm / LM Head", subtitle: "x₂ → vocabulary → loss",
      items: [
        item("p", "final_rms_gamma", "final RMSNorm γ", "parameter"),
        item("a", "final_norm", "final hidden"),
        item("p", "W_lm_head", "lm_head weight", "parameter"),
        item("a", "logits", "vocabulary logits"),
        item("a", "probabilities", "softmax probabilities"),
        item("a", "loss", "cross-entropy loss", "loss"),
      ],
    },
  ];

  const backward = [
    {
      id: "loss", title: "Loss Gradient", subtitle: "loss → dLogits",
      items: [
        item("ga", "logits", "dLogits", "gradient"),
        item("ga", "final_norm", "dFinal hidden", "gradient"),
        item("gp", "W_lm_head", "dW_lm_head", "parameter-gradient"),
        item("gp", "final_rms_gamma", "dFinal γ", "parameter-gradient"),
        item("ga", "residual2", "dResidual x₂", "gradient"),
      ],
    },
    {
      id: "ffn-backward", title: "FFN Gradient", subtitle: "downᵀ → gate/upᵀ → RMSNormᵀ",
      items: [
        item("ga", "residual1_direct", "direct dResidual x₁", "gradient"),
        item("ga", "ffn_out", "dFFN output", "gradient"),
        item("ga", "gated", "dGated hidden", "gradient"),
        item("gp", "W_down", "dWdown", "parameter-gradient"),
        item("ga", "silu_gate", "dSiLU(gate)", "gradient"),
        item("ga", "up", "dUp", "gradient"),
        item("ga", "gate", "dGate", "gradient"),
        item("gp", "W_gate", "dWgate", "parameter-gradient"),
        item("gp", "W_up", "dWup", "parameter-gradient"),
        item("ga", "rms2_gate", "dRMS2 from gate", "gradient"),
        item("ga", "rms2_up", "dRMS2 from up", "gradient"),
        item("ga", "rms2", "accumulated dRMS2", "gradient"),
        item("gp", "rms2_gamma", "dRMS2 γ", "parameter-gradient"),
        item("ga", "residual1_ffn", "FFN path dResidual x₁", "gradient"),
        item("ga", "residual1", "accumulated dResidual x₁", "gradient"),
      ],
    },
    {
      id: "attention-backward", title: "Attention Gradient", subtitle: "Woᵀ → dAV → dQKᵀ",
      items: [
        item("ga", "embedding_direct", "direct dEmbedding", "gradient"),
        item("ga", "attn_out", "dAttention output", "gradient"),
        item("gp", "W_o", "dWo", "parameter-gradient"),
        item("ga", "context", "dContext", "gradient"),
        item("ga", "context_heads", "dContext heads", "gradient"),
        item("ga", "attention_weights", "dAttention weights", "gradient"),
        item("ga", "v_repeated", "dRepeated V", "gradient"),
        item("ga", "causal_scores", "dScores", "gradient"),
        item("ga", "q_rope", "dRoPE(Q)", "gradient"),
        item("ga", "k_repeated", "dRepeated K", "gradient"),
      ],
    },
    {
      id: "qkv-backward", title: "Q / K / V Gradient", subtitle: "GQA reduce → projectionsᵀ",
      items: [
        item("ga", "k_rope", "reduced dK RoPE", "gradient"),
        item("ga", "v_heads", "reduced dV heads", "gradient"),
        item("ga", "q_heads", "inverse-RoPE dQ", "gradient"),
        item("ga", "k_heads", "inverse-RoPE dK", "gradient"),
        item("ga", "q_linear", "dQ projection", "gradient"),
        item("ga", "k_linear", "dK projection", "gradient"),
        item("ga", "v_linear", "dV projection", "gradient"),
        item("gp", "W_q", "dWq", "parameter-gradient"),
        item("gp", "W_k", "dWk", "parameter-gradient"),
        item("gp", "W_v", "dWv", "parameter-gradient"),
        item("ga", "rms1_q", "dRMS1 from Q", "gradient"),
        item("ga", "rms1_k", "dRMS1 from K", "gradient"),
        item("ga", "rms1_v", "dRMS1 from V", "gradient"),
        item("ga", "rms1", "accumulated dRMS1", "gradient"),
        item("gp", "rms1_gamma", "dRMS1 γ", "parameter-gradient"),
        item("ga", "embedding_attention", "attention path dEmbedding", "gradient"),
        item("ga", "embedding", "accumulated dEmbedding", "gradient"),
        item("gp", "embedding_table", "dEmbedding table", "parameter-gradient"),
      ],
    },
  ];

  const groups = Object.freeze({ forward: Object.freeze(forward), backward: Object.freeze(backward) });

  function flatten(mode) {
    if (!groups[mode]) throw new Error(`Unknown stream mode: ${mode}`);
    return groups[mode].flatMap((group) => group.items.map((entry) => ({ ...entry, groupId: group.id, groupTitle: group.title })));
  }

  function keyOf(entry) { return `${entry.source}:${entry.key}`; }

  function coverage(mode, result) {
    const expected = mode === "forward"
      ? { a: Object.keys(result.activations), p: Object.keys(result.parameters) }
      : { ga: Object.keys(result.gradients.activations), gp: Object.keys(result.gradients.parameters) };
    const registered = new Set(flatten(mode).map(keyOf));
    const missing = [];
    const duplicates = [];
    const seen = new Set();
    for (const entry of flatten(mode)) {
      const key = keyOf(entry);
      if (seen.has(key)) duplicates.push(key);
      seen.add(key);
    }
    for (const [source, keys] of Object.entries(expected)) {
      for (const key of keys) if (!registered.has(`${source}:${key}`)) missing.push(`${source}:${key}`);
    }
    const unexpected = [...registered].filter((key) => {
      const [source, keyName] = key.split(":");
      return !expected[source] || !expected[source].includes(keyName);
    });
    return { missing, duplicates, unexpected, complete: missing.length === 0 && duplicates.length === 0 && unexpected.length === 0 };
  }

  return { groups, flatten, coverage, keyOf };
});
