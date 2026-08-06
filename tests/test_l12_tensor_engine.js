const assert = require("node:assert/strict");
const { test } = require("node:test");

const Engine = require("../M2 Transformer与大模型/实践代码/L12_tensor_engine.js");

function allFinite(tensor) {
  return tensor.data.every(Number.isFinite);
}

function numericGradient(result, parameterName, index, epsilon = 1e-5) {
  const plus = Engine.cloneParameters(result.parameters);
  const minus = Engine.cloneParameters(result.parameters);
  plus[parameterName].data[index] += epsilon;
  minus[parameterName].data[index] -= epsilon;
  const plusLoss = Engine.run(result.config, result.seed, plus).loss;
  const minusLoss = Engine.run(result.config, result.seed, minus).loss;
  return (plusLoss - minusLoss) / (2 * epsilon);
}

test("all three presets run a complete finite forward/backward pass", () => {
  for (const [name, preset] of Object.entries(Engine.PRESETS)) {
    const result = Engine.run(preset, 42);
    assert.ok(Number.isFinite(result.loss), `${name}: loss must be finite`);
    assert.ok(result.parameterCount > 0, `${name}: parameter count must be positive`);
    for (const [tensorName, tensor] of Object.entries(result.gradients.activations)) {
      assert.ok(allFinite(tensor), `${name}: activation gradient ${tensorName} contains NaN/Inf`);
    }
    for (const [parameterName, tensor] of Object.entries(result.gradients.parameters)) {
      assert.ok(allFinite(tensor), `${name}: parameter gradient ${parameterName} contains NaN/Inf`);
    }
  }
});

test("causal mask and attention weights have the expected structure", () => {
  const result = Engine.run(Engine.PRESETS.standard, 42);
  const scores = result.activations.causal_scores;
  const weights = result.activations.attention_weights;
  const [batch, heads, sequence] = [scores.shape[0], scores.shape[1], scores.shape[2]];
  for (let b = 0; b < batch; b += 1) {
    for (let h = 0; h < heads; h += 1) {
      for (let i = 0; i < sequence; i += 1) {
        let rowSum = 0;
        for (let j = 0; j < sequence; j += 1) {
          const index = ((b * heads + h) * sequence + i) * sequence + j;
          if (j > i) {
            assert.equal(scores.data[index], -Infinity, `score ${i},${j} must be masked`);
            assert.equal(weights.data[index], 0, `weight ${i},${j} must be zero`);
          } else {
            assert.ok(Number.isFinite(scores.data[index]));
            rowSum += weights.data[index];
          }
        }
        assert.ok(Math.abs(rowSum - 1) < 1e-12, `attention row must sum to 1, got ${rowSum}`);
      }
    }
  }
});

test("GQA repeats K/V in forward and reduces their gradients in backward", () => {
  const result = Engine.run(Engine.PRESETS.standard, 42);
  const k = result.activations.k_rope;
  const repeated = result.activations.k_repeated;
  const reduced = result.gradients.activations.k_rope;
  const [batch, kvHeads, sequence, headDim] = k.shape;
  const queryHeads = repeated.shape[1];
  const repeat = queryHeads / kvHeads;
  for (let b = 0; b < batch; b += 1) {
    for (let queryHead = 0; queryHead < queryHeads; queryHead += 1) {
      const sourceHead = Math.floor(queryHead / repeat);
      for (let s = 0; s < sequence; s += 1) {
        for (let d = 0; d < headDim; d += 1) {
          const sourceIndex = ((b * kvHeads + sourceHead) * sequence + s) * headDim + d;
          const repeatedIndex = ((b * queryHeads + queryHead) * sequence + s) * headDim + d;
          assert.equal(repeated.data[repeatedIndex], k.data[sourceIndex]);
        }
      }
    }
  }
  assert.equal(reduced.shape[1], kvHeads);
  assert.ok(allFinite(reduced));
});

test("reverse-mode gradients agree with finite differences", () => {
  const result = Engine.run(Engine.PRESETS.compact, 42);
  const checks = [
    ["W_q", 3], ["W_k", 4], ["W_v", 5], ["W_o", 6],
    ["W_gate", 7], ["W_up", 8], ["W_down", 9], ["W_lm_head", 10],
    ["rms1_gamma", 2], ["rms2_gamma", 3], ["final_rms_gamma", 4],
    ["embedding_table", 17],
  ];
  for (const [name, index] of checks) {
    const numerical = numericGradient(result, name, index);
    const analytical = result.gradients.parameters[name].data[index];
    const relativeError = Math.abs(numerical - analytical) / Math.max(1e-8, Math.abs(numerical) + Math.abs(analytical));
    assert.ok(relativeError < 1e-5, `${name}[${index}] relative error ${relativeError}`);
  }
});

test("one default SGD step lowers loss for every preset", () => {
  for (const [name, preset] of Object.entries(Engine.PRESETS)) {
    const result = Engine.run(preset, 42);
    const update = Engine.sgdStep(result, 0.05);
    assert.ok(update.after.loss < update.before.loss, `${name}: SGD did not lower loss`);
    assert.equal(update.after.parameterCount, update.before.parameterCount);
  }
});
