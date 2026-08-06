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

  const tensorId = (reference) => `tensor_${reference.replace(":", "_")}`;
  const tensorNode = (reference, x, y) => ({ id: tensorId(reference), type: "tensor", reference, x, y });
  const operatorNode = (id, label, formula, x, y) => ({ id: `op_${id}`, type: "operator", label, formula, x, y });
  const edge = (from, to, options = {}) => ({ from, to, ...options });
  const tensorEdge = (fromReference, to, options) => edge(tensorId(fromReference), `op_${to}`, options);
  const outputEdge = (from, toReference, options) => edge(`op_${from}`, tensorId(toReference), options);
  const operationEdge = (from, to, options) => edge(`op_${from}`, `op_${to}`, options);

  const forwardGraph = {
    width: 9620,
    height: 1660,
    sections: [
      { x: 0, width: 1040, label: "Input / Embedding" },
      { x: 1040, width: 2020, label: "QKV / RoPE / GQA" },
      { x: 3060, width: 1500, label: "Causal Attention" },
      { x: 4560, width: 1240, label: "Attention Residual" },
      { x: 5800, width: 2000, label: "SwiGLU FFN" },
      { x: 7800, width: 1820, label: "Output / Loss" },
    ],
    nodes: [
      tensorNode("a:token_ids", 40, 650), tensorNode("a:target_ids", 40, 1350), tensorNode("p:embedding_table", 40, 250),
      operatorNode("embedding", "Embedding Lookup", "X = E[token_ids]", 280, 680), tensorNode("a:embedding", 500, 650),
      tensorNode("p:rms1_gamma", 560, 250), operatorNode("rms1", "RMSNorm", "x / RMS(x) ⊙ γ", 720, 680), tensorNode("a:rms1", 940, 650),
      tensorNode("p:W_q", 1040, 80), tensorNode("p:W_k", 1040, 250), tensorNode("p:W_v", 1040, 420),
      operatorNode("qkv", "Q / K / V Linear", "XWq · XWk · XWv", 1320, 680),
      tensorNode("a:q_linear", 1540, 350), tensorNode("a:k_linear", 1540, 620), tensorNode("a:v_linear", 1540, 890),
      operatorNode("split", "Split Heads", "[B,S,width] → [B,h,S,d_head]", 1780, 680),
      tensorNode("a:q_heads", 2000, 350), tensorNode("a:k_heads", 2000, 620), tensorNode("a:v_heads", 2000, 890),
      operatorNode("rope_q", "RoPE(Q)", "pairwise rotation", 2240, 380), operatorNode("rope_k", "RoPE(K)", "pairwise rotation", 2240, 650),
      tensorNode("a:q_rope", 2440, 350), tensorNode("a:k_rope", 2440, 620),
      operatorNode("gqa_k", "GQA Broadcast K", "h_kv → h", 2660, 650), operatorNode("gqa_v", "GQA Broadcast V", "h_kv → h", 2660, 920),
      tensorNode("a:k_repeated", 2880, 620), tensorNode("a:v_repeated", 2880, 890),
      operatorNode("qk", "Scaled QKᵀ + Mask", "QKᵀ / √d_head + M", 3100, 500), tensorNode("a:causal_scores", 3320, 470),
      operatorNode("attn_softmax", "Row Softmax", "softmax(scores)", 3520, 500), tensorNode("a:attention_weights", 3740, 470),
      operatorNode("av", "Weighted Value", "A · V", 3940, 680), tensorNode("a:context_heads", 4160, 650),
      operatorNode("concat", "Concat Heads", "[B,h,S,d_head] → [B,S,d]", 4360, 680), tensorNode("a:context", 4560, 650),
      tensorNode("p:W_o", 4560, 250), operatorNode("wo", "Output Linear", "context · Wo", 4760, 680), tensorNode("a:attn_out", 4960, 650),
      operatorNode("add1", "Residual Add 1", "x₁ = x₀ + attention", 5160, 680), tensorNode("a:residual1", 5360, 650),
      tensorNode("p:rms2_gamma", 5360, 250), operatorNode("rms2", "RMSNorm", "x₁ / RMS(x₁) ⊙ γ", 5560, 680), tensorNode("a:rms2", 5760, 650),
      tensorNode("p:W_gate", 5800, 120), tensorNode("p:W_up", 5800, 300), operatorNode("gate_up", "Gate / Up Linear", "xWgate · xWup", 6000, 700),
      tensorNode("a:gate", 6220, 520), tensorNode("a:up", 6220, 820), operatorNode("silu", "SiLU", "g · sigmoid(g)", 6440, 550),
      tensorNode("a:silu_gate", 6600, 520), operatorNode("hadamard", "Hadamard Product", "SiLU(gate) ⊙ up", 6800, 680), tensorNode("a:gated", 7000, 650),
      tensorNode("p:W_down", 7000, 250), operatorNode("down", "Down Linear", "gated · Wdown", 7200, 680), tensorNode("a:ffn_out", 7400, 650),
      operatorNode("add2", "Residual Add 2", "x₂ = x₁ + FFN", 7600, 680), tensorNode("a:residual2", 7800, 650),
      tensorNode("p:final_rms_gamma", 7800, 250), operatorNode("final_rms", "Final RMSNorm", "x₂ / RMS(x₂) ⊙ γ", 8000, 680), tensorNode("a:final_norm", 8200, 650),
      tensorNode("p:W_lm_head", 8200, 250), operatorNode("lm_head", "LM Head Linear", "hidden · Wlm", 8400, 680), tensorNode("a:logits", 8600, 650),
      operatorNode("output_softmax", "Vocabulary Softmax", "softmax(logits)", 8800, 500), tensorNode("a:probabilities", 9000, 470),
      operatorNode("cross_entropy", "Cross Entropy", "−mean log p[target]", 9200, 680), tensorNode("a:loss", 9400, 650),
    ],
    edges: [
      tensorEdge("a:token_ids", "embedding"), tensorEdge("p:embedding_table", "embedding"), outputEdge("embedding", "a:embedding"),
      tensorEdge("a:embedding", "rms1"), tensorEdge("p:rms1_gamma", "rms1"), outputEdge("rms1", "a:rms1"),
      tensorEdge("a:rms1", "qkv"), tensorEdge("p:W_q", "qkv"), tensorEdge("p:W_k", "qkv"), tensorEdge("p:W_v", "qkv"),
      outputEdge("qkv", "a:q_linear"), outputEdge("qkv", "a:k_linear"), outputEdge("qkv", "a:v_linear"),
      tensorEdge("a:q_linear", "split"), tensorEdge("a:k_linear", "split"), tensorEdge("a:v_linear", "split"),
      outputEdge("split", "a:q_heads"), outputEdge("split", "a:k_heads"), outputEdge("split", "a:v_heads"),
      tensorEdge("a:q_heads", "rope_q"), outputEdge("rope_q", "a:q_rope"), tensorEdge("a:k_heads", "rope_k"), outputEdge("rope_k", "a:k_rope"),
      tensorEdge("a:k_rope", "gqa_k"), outputEdge("gqa_k", "a:k_repeated"), tensorEdge("a:v_heads", "gqa_v"), outputEdge("gqa_v", "a:v_repeated"),
      tensorEdge("a:q_rope", "qk"), tensorEdge("a:k_repeated", "qk"), outputEdge("qk", "a:causal_scores"),
      tensorEdge("a:causal_scores", "attn_softmax"), outputEdge("attn_softmax", "a:attention_weights"),
      tensorEdge("a:attention_weights", "av"), tensorEdge("a:v_repeated", "av"), outputEdge("av", "a:context_heads"),
      tensorEdge("a:context_heads", "concat"), outputEdge("concat", "a:context"), tensorEdge("a:context", "wo"), tensorEdge("p:W_o", "wo"), outputEdge("wo", "a:attn_out"),
      tensorEdge("a:attn_out", "add1"), tensorEdge("a:embedding", "add1", { routeY: 1220, kind: "residual" }), outputEdge("add1", "a:residual1"),
      tensorEdge("a:residual1", "rms2"), tensorEdge("p:rms2_gamma", "rms2"), outputEdge("rms2", "a:rms2"),
      tensorEdge("a:rms2", "gate_up"), tensorEdge("p:W_gate", "gate_up"), tensorEdge("p:W_up", "gate_up"), outputEdge("gate_up", "a:gate"), outputEdge("gate_up", "a:up"),
      tensorEdge("a:gate", "silu"), outputEdge("silu", "a:silu_gate"), tensorEdge("a:silu_gate", "hadamard"), tensorEdge("a:up", "hadamard"), outputEdge("hadamard", "a:gated"),
      tensorEdge("a:gated", "down"), tensorEdge("p:W_down", "down"), outputEdge("down", "a:ffn_out"),
      tensorEdge("a:ffn_out", "add2"), tensorEdge("a:residual1", "add2", { routeY: 1280, kind: "residual" }), outputEdge("add2", "a:residual2"),
      tensorEdge("a:residual2", "final_rms"), tensorEdge("p:final_rms_gamma", "final_rms"), outputEdge("final_rms", "a:final_norm"),
      tensorEdge("a:final_norm", "lm_head"), tensorEdge("p:W_lm_head", "lm_head"), outputEdge("lm_head", "a:logits"),
      tensorEdge("a:logits", "output_softmax"), outputEdge("output_softmax", "a:probabilities"), tensorEdge("a:probabilities", "cross_entropy"),
      tensorEdge("a:target_ids", "cross_entropy", { routeY: 1480, kind: "target" }), outputEdge("cross_entropy", "a:loss"),
    ],
  };

  const backwardGraph = {
    width: 10840,
    height: 1780,
    sections: [
      { x: 0, width: 1160, label: "Loss / Output Gradient" },
      { x: 1160, width: 3300, label: "FFN / Residual Gradient" },
      { x: 4460, width: 2640, label: "Attention Gradient" },
      { x: 7100, width: 2200, label: "QKV Gradient" },
      { x: 9300, width: 1540, label: "Input / Embedding Gradient" },
    ],
    nodes: [
      tensorNode("ga:logits", 40, 700), operatorNode("lm_bwd", "LM Head Backward", "dH=dZ·Wᵀ; dW=Hᵀ·dZ", 280, 730),
      tensorNode("ga:final_norm", 500, 600), tensorNode("gp:W_lm_head", 500, 250), operatorNode("final_rms_bwd", "Final RMSNorm Backward", "chain rule through RMS", 720, 630),
      tensorNode("ga:residual2", 940, 600), tensorNode("gp:final_rms_gamma", 940, 250), operatorNode("split2_bwd", "Residual 2 Split", "copy upstream gradient", 1160, 630),
      tensorNode("ga:residual1_direct", 1380, 350), tensorNode("ga:ffn_out", 1380, 800), operatorNode("down_bwd", "Down Linear Backward", "dX=dY·Wᵀ; dW=Xᵀ·dY", 1600, 830),
      tensorNode("ga:gated", 1820, 800), tensorNode("gp:W_down", 1820, 1100), operatorNode("mul_bwd", "Hadamard Backward", "dA=dY⊙B; dB=dY⊙A", 2040, 830),
      tensorNode("ga:silu_gate", 2260, 650), tensorNode("ga:up", 2260, 950), operatorNode("silu_bwd", "SiLU Backward", "dGate=dSiLU⊙SiLU′", 2480, 680), tensorNode("ga:gate", 2700, 650),
      operatorNode("gate_up_bwd", "Gate / Up Linear Backward", "two projection branches", 2920, 830),
      tensorNode("gp:W_gate", 3140, 260), tensorNode("gp:W_up", 3140, 440), tensorNode("ga:rms2_gate", 3140, 680), tensorNode("ga:rms2_up", 3140, 950),
      operatorNode("sum_rms2", "Accumulate", "dRMS2_gate + dRMS2_up", 3360, 830), tensorNode("ga:rms2", 3580, 800),
      operatorNode("rms2_bwd", "RMSNorm 2 Backward", "input gradient + dγ", 3800, 830), tensorNode("ga:residual1_ffn", 4020, 800), tensorNode("gp:rms2_gamma", 4020, 1100),
      operatorNode("sum_residual1", "Accumulate Residual 1", "direct + FFN path", 4240, 630), tensorNode("ga:residual1", 4460, 600),
      operatorNode("split1_bwd", "Residual 1 Split", "copy upstream gradient", 4680, 630), tensorNode("ga:embedding_direct", 4900, 350), tensorNode("ga:attn_out", 4900, 800),
      operatorNode("wo_bwd", "Output Linear Backward", "dContext / dWo", 5120, 830), tensorNode("ga:context", 5340, 800), tensorNode("gp:W_o", 5340, 1100),
      operatorNode("split_heads_bwd", "Split Head Gradient", "[B,S,d] → [B,h,S,d_head]", 5560, 830), tensorNode("ga:context_heads", 5780, 800),
      operatorNode("av_bwd", "A·V Backward", "dA=dC·Vᵀ; dV=Aᵀ·dC", 6000, 830), tensorNode("ga:attention_weights", 6220, 650), tensorNode("ga:v_repeated", 6220, 950),
      operatorNode("softmax_bwd", "Softmax Backward", "A⊙(dA−sum(dA⊙A))", 6440, 680), tensorNode("ga:causal_scores", 6660, 650),
      operatorNode("qk_bwd", "QKᵀ Backward", "dQ=dS·K; dK=dSᵀ·Q", 6880, 680), tensorNode("ga:q_rope", 7100, 500), tensorNode("ga:k_repeated", 7100, 800),
      operatorNode("rope_q_bwd", "RoPE(Q) Backward", "inverse rotation", 7320, 530), operatorNode("gqa_k_bwd", "GQA Reduce K", "sum shared-head gradients", 7320, 830),
      operatorNode("gqa_v_bwd", "GQA Reduce V", "sum shared-head gradients", 7320, 1080), tensorNode("ga:q_heads", 7540, 500), tensorNode("ga:k_rope", 7540, 800), tensorNode("ga:v_heads", 7540, 1050),
      operatorNode("rope_k_bwd", "RoPE(K) Backward", "inverse rotation", 7760, 830), operatorNode("merge_q_bwd", "Merge dQ Heads", "head layout → linear layout", 7760, 530),
      operatorNode("merge_v_bwd", "Merge dV Heads", "head layout → linear layout", 7760, 1080), tensorNode("ga:q_linear", 7980, 500), tensorNode("ga:k_heads", 7980, 800), tensorNode("ga:v_linear", 7980, 1050),
      operatorNode("merge_k_bwd", "Merge dK Heads", "head layout → linear layout", 8200, 830), operatorNode("q_proj_bwd", "Q Projection Backward", "dX / dWq", 8200, 530),
      operatorNode("v_proj_bwd", "V Projection Backward", "dX / dWv", 8200, 1080), tensorNode("ga:rms1_q", 8420, 420), tensorNode("gp:W_q", 8420, 220),
      tensorNode("ga:k_linear", 8420, 800), tensorNode("ga:rms1_v", 8420, 1120), tensorNode("gp:W_v", 8420, 1370),
      operatorNode("k_proj_bwd", "K Projection Backward", "dX / dWk", 8640, 830), tensorNode("ga:rms1_k", 8860, 700), tensorNode("gp:W_k", 8860, 1020),
      operatorNode("sum_rms1", "Accumulate Q/K/V", "dXq + dXk + dXv", 9080, 780), tensorNode("ga:rms1", 9300, 750),
      operatorNode("rms1_bwd", "RMSNorm 1 Backward", "attention path + dγ", 9520, 780), tensorNode("ga:embedding_attention", 9740, 650), tensorNode("gp:rms1_gamma", 9740, 1000),
      operatorNode("sum_embedding", "Accumulate Embedding", "direct + attention path", 9960, 530), tensorNode("ga:embedding", 10180, 500),
      operatorNode("embedding_bwd", "Embedding Scatter-Add", "dE[token] += dX", 10400, 530), tensorNode("gp:embedding_table", 10620, 500),
    ],
    edges: [
      tensorEdge("ga:logits", "lm_bwd"), outputEdge("lm_bwd", "ga:final_norm"), outputEdge("lm_bwd", "gp:W_lm_head"),
      tensorEdge("ga:final_norm", "final_rms_bwd"), outputEdge("final_rms_bwd", "ga:residual2"), outputEdge("final_rms_bwd", "gp:final_rms_gamma"),
      tensorEdge("ga:residual2", "split2_bwd"), outputEdge("split2_bwd", "ga:residual1_direct"), outputEdge("split2_bwd", "ga:ffn_out"),
      tensorEdge("ga:ffn_out", "down_bwd"), outputEdge("down_bwd", "ga:gated"), outputEdge("down_bwd", "gp:W_down"),
      tensorEdge("ga:gated", "mul_bwd"), outputEdge("mul_bwd", "ga:silu_gate"), outputEdge("mul_bwd", "ga:up"), tensorEdge("ga:silu_gate", "silu_bwd"), outputEdge("silu_bwd", "ga:gate"),
      tensorEdge("ga:gate", "gate_up_bwd"), tensorEdge("ga:up", "gate_up_bwd"), outputEdge("gate_up_bwd", "gp:W_gate"), outputEdge("gate_up_bwd", "gp:W_up"),
      outputEdge("gate_up_bwd", "ga:rms2_gate"), outputEdge("gate_up_bwd", "ga:rms2_up"), tensorEdge("ga:rms2_gate", "sum_rms2"), tensorEdge("ga:rms2_up", "sum_rms2"), outputEdge("sum_rms2", "ga:rms2"),
      tensorEdge("ga:rms2", "rms2_bwd"), outputEdge("rms2_bwd", "ga:residual1_ffn"), outputEdge("rms2_bwd", "gp:rms2_gamma"),
      tensorEdge("ga:residual1_direct", "sum_residual1", { routeY: 120, kind: "residual" }), tensorEdge("ga:residual1_ffn", "sum_residual1"), outputEdge("sum_residual1", "ga:residual1"),
      tensorEdge("ga:residual1", "split1_bwd"), outputEdge("split1_bwd", "ga:embedding_direct"), outputEdge("split1_bwd", "ga:attn_out"),
      tensorEdge("ga:attn_out", "wo_bwd"), outputEdge("wo_bwd", "ga:context"), outputEdge("wo_bwd", "gp:W_o"), tensorEdge("ga:context", "split_heads_bwd"), outputEdge("split_heads_bwd", "ga:context_heads"),
      tensorEdge("ga:context_heads", "av_bwd"), outputEdge("av_bwd", "ga:attention_weights"), outputEdge("av_bwd", "ga:v_repeated"), tensorEdge("ga:attention_weights", "softmax_bwd"), outputEdge("softmax_bwd", "ga:causal_scores"),
      tensorEdge("ga:causal_scores", "qk_bwd"), outputEdge("qk_bwd", "ga:q_rope"), outputEdge("qk_bwd", "ga:k_repeated"),
      tensorEdge("ga:q_rope", "rope_q_bwd"), outputEdge("rope_q_bwd", "ga:q_heads"), tensorEdge("ga:k_repeated", "gqa_k_bwd"), outputEdge("gqa_k_bwd", "ga:k_rope"),
      tensorEdge("ga:v_repeated", "gqa_v_bwd"), outputEdge("gqa_v_bwd", "ga:v_heads"), tensorEdge("ga:k_rope", "rope_k_bwd"), outputEdge("rope_k_bwd", "ga:k_heads"),
      tensorEdge("ga:q_heads", "merge_q_bwd"), outputEdge("merge_q_bwd", "ga:q_linear"), tensorEdge("ga:k_heads", "merge_k_bwd"), outputEdge("merge_k_bwd", "ga:k_linear"),
      tensorEdge("ga:v_heads", "merge_v_bwd"), outputEdge("merge_v_bwd", "ga:v_linear"),
      tensorEdge("ga:q_linear", "q_proj_bwd"), outputEdge("q_proj_bwd", "ga:rms1_q"), outputEdge("q_proj_bwd", "gp:W_q"),
      tensorEdge("ga:k_linear", "k_proj_bwd"), outputEdge("k_proj_bwd", "ga:rms1_k"), outputEdge("k_proj_bwd", "gp:W_k"),
      tensorEdge("ga:v_linear", "v_proj_bwd"), outputEdge("v_proj_bwd", "ga:rms1_v"), outputEdge("v_proj_bwd", "gp:W_v"),
      tensorEdge("ga:rms1_q", "sum_rms1"), tensorEdge("ga:rms1_k", "sum_rms1"), tensorEdge("ga:rms1_v", "sum_rms1"), outputEdge("sum_rms1", "ga:rms1"),
      tensorEdge("ga:rms1", "rms1_bwd"), outputEdge("rms1_bwd", "ga:embedding_attention"), outputEdge("rms1_bwd", "gp:rms1_gamma"),
      tensorEdge("ga:embedding_direct", "sum_embedding", { routeY: 120, kind: "residual" }), tensorEdge("ga:embedding_attention", "sum_embedding"), outputEdge("sum_embedding", "ga:embedding"),
      tensorEdge("ga:embedding", "embedding_bwd"), outputEdge("embedding_bwd", "gp:embedding_table"),
    ],
  };

  const graphs = Object.freeze({ forward: forwardGraph, backward: backwardGraph });

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

  function graphIntegrity(mode) {
    const graph = graphs[mode];
    if (!graph) throw new Error(`Unknown graph mode: ${mode}`);
    const nodeIds = new Set();
    const duplicateNodeIds = [];
    for (const node of graph.nodes) {
      if (nodeIds.has(node.id)) duplicateNodeIds.push(node.id);
      nodeIds.add(node.id);
    }
    const danglingEdges = graph.edges.filter((connection) => !nodeIds.has(connection.from) || !nodeIds.has(connection.to));
    const graphTensorReferences = graph.nodes.filter((node) => node.type === "tensor").map((node) => node.reference);
    const registeredReferences = flatten(mode).map(keyOf);
    const graphSet = new Set(graphTensorReferences);
    const missingTensors = registeredReferences.filter((reference) => !graphSet.has(reference));
    const unexpectedTensors = graphTensorReferences.filter((reference) => !registeredReferences.includes(reference));
    const duplicateTensors = graphTensorReferences.filter((reference, index) => graphTensorReferences.indexOf(reference) !== index);
    return {
      duplicateNodeIds, danglingEdges, missingTensors, unexpectedTensors, duplicateTensors,
      complete: duplicateNodeIds.length === 0 && danglingEdges.length === 0 && missingTensors.length === 0 && unexpectedTensors.length === 0 && duplicateTensors.length === 0,
    };
  }

  return { groups, graphs, flatten, coverage, graphIntegrity, keyOf, tensorId };
});
