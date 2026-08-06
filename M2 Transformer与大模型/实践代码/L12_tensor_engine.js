/*
 * L12 Transformer matrix-flow lab: deterministic, dependency-free numerical core.
 *
 * The browser UI and Node tests both use this file. Every forward value and
 * backward gradient is computed with Float64Array; rounding happens only in the UI.
 */
(function attachL12Engine(root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  root.L12Engine = api;
})(typeof globalThis !== "undefined" ? globalThis : this, function buildL12Engine() {
  "use strict";

  const PRESETS = Object.freeze({
    compact: Object.freeze({
      id: "compact", label: "紧凑", batch: 1, seq: 6, hidden: 16,
      heads: 4, kvHeads: 2, intermediate: 56, vocab: 32,
      ropeTheta: 500000, epsilon: 1e-5,
    }),
    standard: Object.freeze({
      id: "standard", label: "标准", batch: 2, seq: 8, hidden: 32,
      heads: 4, kvHeads: 2, intermediate: 112, vocab: 64,
      ropeTheta: 500000, epsilon: 1e-5,
    }),
    expanded: Object.freeze({
      id: "expanded", label: "扩展", batch: 4, seq: 16, hidden: 64,
      heads: 8, kvHeads: 2, intermediate: 224, vocab: 128,
      ropeTheta: 500000, epsilon: 1e-5,
    }),
  });

  function sizeOf(shape) {
    return shape.reduce((product, value) => product * value, 1);
  }

  function makeTensor(name, shape, data) {
    const expected = sizeOf(shape);
    const values = data instanceof Float64Array ? data : new Float64Array(data || expected);
    if (values.length !== expected) {
      throw new Error(`${name}: data length ${values.length} != shape size ${expected}`);
    }
    return { name, shape: shape.slice(), data: values };
  }

  function zeros(name, shape) {
    return makeTensor(name, shape, new Float64Array(sizeOf(shape)));
  }

  function cloneTensor(source, name = source.name) {
    return makeTensor(name, source.shape, new Float64Array(source.data));
  }

  function sameShape(a, b, operation) {
    if (a.data.length !== b.data.length || a.shape.join(",") !== b.shape.join(",")) {
      throw new Error(`${operation}: incompatible shapes [${a.shape}] and [${b.shape}]`);
    }
  }

  function addTensors(name, a, b) {
    sameShape(a, b, name);
    const output = zeros(name, a.shape);
    for (let index = 0; index < output.data.length; index += 1) {
      output.data[index] = a.data[index] + b.data[index];
    }
    return output;
  }

  function addInPlace(target, source) {
    sameShape(target, source, "addInPlace");
    for (let index = 0; index < target.data.length; index += 1) {
      target.data[index] += source.data[index];
    }
    return target;
  }

  function multiplyTensors(name, a, b) {
    sameShape(a, b, name);
    const output = zeros(name, a.shape);
    for (let index = 0; index < output.data.length; index += 1) {
      output.data[index] = a.data[index] * b.data[index];
    }
    return output;
  }

  function makeRng(seed) {
    let state = (Number(seed) >>> 0) || 1;
    let spare = null;
    function uniform() {
      state = (state + 0x6d2b79f5) >>> 0;
      let value = state;
      value = Math.imul(value ^ (value >>> 15), value | 1);
      value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
      return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
    }
    return {
      normal() {
        if (spare !== null) {
          const value = spare;
          spare = null;
          return value;
        }
        const u1 = Math.max(uniform(), 1e-12);
        const u2 = uniform();
        const radius = Math.sqrt(-2 * Math.log(u1));
        const angle = 2 * Math.PI * u2;
        spare = radius * Math.sin(angle);
        return radius * Math.cos(angle);
      },
    };
  }

  function validateConfig(config) {
    const keys = ["batch", "seq", "hidden", "heads", "kvHeads", "intermediate", "vocab"];
    for (const key of keys) {
      if (!Number.isInteger(config[key]) || config[key] < 1) {
        throw new Error(`${key} must be a positive integer`);
      }
    }
    if (config.hidden % config.heads !== 0) throw new Error("hidden must divide evenly into heads");
    if (config.heads % config.kvHeads !== 0) throw new Error("heads must divide evenly into kvHeads");
    if ((config.hidden / config.heads) % 2 !== 0) throw new Error("RoPE requires an even head dimension");
    return config;
  }

  function initializeParameters(config, seed) {
    const rng = makeRng(seed);
    const { hidden: d, intermediate: ff, vocab: V, heads: h, kvHeads: hk } = config;
    const headDim = d / h;
    const kvWidth = hk * headDim;
    const parameters = {};

    function randomParameter(name, shape, scale) {
      const tensor = zeros(name, shape);
      for (let index = 0; index < tensor.data.length; index += 1) {
        tensor.data[index] = rng.normal() * scale;
      }
      parameters[name] = tensor;
    }

    function gammaParameter(name) {
      const tensor = zeros(name, [d]);
      for (let index = 0; index < d; index += 1) tensor.data[index] = 1 + rng.normal() * 0.02;
      parameters[name] = tensor;
    }

    randomParameter("embedding_table", [V, d], 0.32);
    gammaParameter("rms1_gamma");
    randomParameter("W_q", [d, d], 0.8 / Math.sqrt(d));
    randomParameter("W_k", [d, kvWidth], 0.8 / Math.sqrt(d));
    randomParameter("W_v", [d, kvWidth], 0.8 / Math.sqrt(d));
    randomParameter("W_o", [d, d], 0.8 / Math.sqrt(d));
    gammaParameter("rms2_gamma");
    randomParameter("W_gate", [d, ff], 0.8 / Math.sqrt(d));
    randomParameter("W_up", [d, ff], 0.8 / Math.sqrt(d));
    randomParameter("W_down", [ff, d], 0.8 / Math.sqrt(ff));
    gammaParameter("final_rms_gamma");
    randomParameter("W_lm_head", [d, V], 0.8 / Math.sqrt(d));
    return parameters;
  }

  function cloneParameters(parameters) {
    const cloned = {};
    for (const [name, tensor] of Object.entries(parameters)) cloned[name] = cloneTensor(tensor, name);
    return cloned;
  }

  function makeTokens(config) {
    const tokens = zeros("token_ids", [config.batch, config.seq]);
    const targets = zeros("target_ids", [config.batch, config.seq]);
    for (let batch = 0; batch < config.batch; batch += 1) {
      for (let position = 0; position < config.seq; position += 1) {
        const offset = batch * 17 + position * 3;
        tokens.data[batch * config.seq + position] = 1 + (offset % (config.vocab - 1));
        targets.data[batch * config.seq + position] = 1 + ((offset + 3) % (config.vocab - 1));
      }
    }
    return { tokens, targets };
  }

  function embeddingForward(tokens, table, hidden) {
    const [batch, seq] = tokens.shape;
    const output = zeros("embedding", [batch, seq, hidden]);
    for (let b = 0; b < batch; b += 1) {
      for (let s = 0; s < seq; s += 1) {
        const token = tokens.data[b * seq + s];
        for (let d = 0; d < hidden; d += 1) {
          output.data[(b * seq + s) * hidden + d] = table.data[token * hidden + d];
        }
      }
    }
    return output;
  }

  function embeddingBackward(tokens, gradient, tableShape) {
    const tableGradient = zeros("∂L/∂embedding_table", tableShape);
    const [batch, seq] = tokens.shape;
    const hidden = gradient.shape[2];
    for (let b = 0; b < batch; b += 1) {
      for (let s = 0; s < seq; s += 1) {
        const token = tokens.data[b * seq + s];
        for (let d = 0; d < hidden; d += 1) {
          tableGradient.data[token * hidden + d] += gradient.data[(b * seq + s) * hidden + d];
        }
      }
    }
    return tableGradient;
  }

  function linearForward(name, input, weight) {
    const inputWidth = input.shape[input.shape.length - 1];
    if (weight.shape[0] !== inputWidth) throw new Error(`${name}: linear input mismatch`);
    const outputWidth = weight.shape[1];
    const rows = input.data.length / inputWidth;
    const outputShape = input.shape.slice(0, -1).concat(outputWidth);
    const output = zeros(name, outputShape);
    for (let row = 0; row < rows; row += 1) {
      const inputBase = row * inputWidth;
      const outputBase = row * outputWidth;
      for (let inner = 0; inner < inputWidth; inner += 1) {
        const value = input.data[inputBase + inner];
        const weightBase = inner * outputWidth;
        for (let column = 0; column < outputWidth; column += 1) {
          output.data[outputBase + column] += value * weight.data[weightBase + column];
        }
      }
    }
    return output;
  }

  function linearBackward(input, weight, outputGradient, inputGradientName, weightGradientName) {
    const inputWidth = input.shape[input.shape.length - 1];
    const outputWidth = weight.shape[1];
    const rows = input.data.length / inputWidth;
    const inputGradient = zeros(inputGradientName, input.shape);
    const weightGradient = zeros(weightGradientName, weight.shape);
    for (let row = 0; row < rows; row += 1) {
      const inputBase = row * inputWidth;
      const outputBase = row * outputWidth;
      for (let inner = 0; inner < inputWidth; inner += 1) {
        const inputValue = input.data[inputBase + inner];
        const weightBase = inner * outputWidth;
        let inputSum = 0;
        for (let column = 0; column < outputWidth; column += 1) {
          const upstream = outputGradient.data[outputBase + column];
          inputSum += upstream * weight.data[weightBase + column];
          weightGradient.data[weightBase + column] += inputValue * upstream;
        }
        inputGradient.data[inputBase + inner] += inputSum;
      }
    }
    return { inputGradient, weightGradient };
  }

  function rmsNormForward(name, input, gamma, epsilon) {
    const hidden = input.shape[input.shape.length - 1];
    const rows = input.data.length / hidden;
    const output = zeros(name, input.shape);
    const inverseRms = new Float64Array(rows);
    for (let row = 0; row < rows; row += 1) {
      const base = row * hidden;
      let sumSquares = 0;
      for (let d = 0; d < hidden; d += 1) {
        const value = input.data[base + d];
        sumSquares += value * value;
      }
      const inverse = 1 / Math.sqrt(sumSquares / hidden + epsilon);
      inverseRms[row] = inverse;
      for (let d = 0; d < hidden; d += 1) {
        output.data[base + d] = input.data[base + d] * inverse * gamma.data[d];
      }
    }
    return { output, inverseRms };
  }

  function rmsNormBackward(input, gamma, inverseRms, outputGradient, inputGradientName, gammaGradientName) {
    const hidden = input.shape[input.shape.length - 1];
    const rows = input.data.length / hidden;
    const inputGradient = zeros(inputGradientName, input.shape);
    const gammaGradient = zeros(gammaGradientName, gamma.shape);
    for (let row = 0; row < rows; row += 1) {
      const base = row * hidden;
      const inverse = inverseRms[row];
      let dot = 0;
      for (let d = 0; d < hidden; d += 1) {
        dot += outputGradient.data[base + d] * gamma.data[d] * input.data[base + d];
        gammaGradient.data[d] += outputGradient.data[base + d] * input.data[base + d] * inverse;
      }
      const correctionScale = inverse * inverse * inverse * dot / hidden;
      for (let d = 0; d < hidden; d += 1) {
        inputGradient.data[base + d] = (
          outputGradient.data[base + d] * gamma.data[d] * inverse
          - input.data[base + d] * correctionScale
        );
      }
    }
    return { inputGradient, gammaGradient };
  }

  function splitHeads(name, input, headCount, headDim) {
    const [batch, seq, width] = input.shape;
    if (width !== headCount * headDim) throw new Error(`${name}: head split mismatch`);
    const output = zeros(name, [batch, headCount, seq, headDim]);
    for (let b = 0; b < batch; b += 1) {
      for (let s = 0; s < seq; s += 1) {
        for (let h = 0; h < headCount; h += 1) {
          for (let k = 0; k < headDim; k += 1) {
            const inputIndex = (b * seq + s) * width + h * headDim + k;
            const outputIndex = ((b * headCount + h) * seq + s) * headDim + k;
            output.data[outputIndex] = input.data[inputIndex];
          }
        }
      }
    }
    return output;
  }

  function mergeHeads(name, input) {
    const [batch, heads, seq, headDim] = input.shape;
    const width = heads * headDim;
    const output = zeros(name, [batch, seq, width]);
    for (let b = 0; b < batch; b += 1) {
      for (let h = 0; h < heads; h += 1) {
        for (let s = 0; s < seq; s += 1) {
          for (let k = 0; k < headDim; k += 1) {
            const inputIndex = ((b * heads + h) * seq + s) * headDim + k;
            const outputIndex = (b * seq + s) * width + h * headDim + k;
            output.data[outputIndex] = input.data[inputIndex];
          }
        }
      }
    }
    return output;
  }

  function ropeForward(name, input, theta) {
    const [batch, heads, seq, headDim] = input.shape;
    const output = zeros(name, input.shape);
    for (let b = 0; b < batch; b += 1) {
      for (let h = 0; h < heads; h += 1) {
        for (let s = 0; s < seq; s += 1) {
          const base = ((b * heads + h) * seq + s) * headDim;
          for (let pair = 0; pair < headDim; pair += 2) {
            const inverseFrequency = Math.pow(theta, -pair / headDim);
            const angle = s * inverseFrequency;
            const cosine = Math.cos(angle);
            const sine = Math.sin(angle);
            const x0 = input.data[base + pair];
            const x1 = input.data[base + pair + 1];
            output.data[base + pair] = x0 * cosine - x1 * sine;
            output.data[base + pair + 1] = x0 * sine + x1 * cosine;
          }
        }
      }
    }
    return output;
  }

  function ropeBackward(name, outputGradient, theta) {
    const [batch, heads, seq, headDim] = outputGradient.shape;
    const inputGradient = zeros(name, outputGradient.shape);
    for (let b = 0; b < batch; b += 1) {
      for (let h = 0; h < heads; h += 1) {
        for (let s = 0; s < seq; s += 1) {
          const base = ((b * heads + h) * seq + s) * headDim;
          for (let pair = 0; pair < headDim; pair += 2) {
            const inverseFrequency = Math.pow(theta, -pair / headDim);
            const angle = s * inverseFrequency;
            const cosine = Math.cos(angle);
            const sine = Math.sin(angle);
            const y0 = outputGradient.data[base + pair];
            const y1 = outputGradient.data[base + pair + 1];
            inputGradient.data[base + pair] = y0 * cosine + y1 * sine;
            inputGradient.data[base + pair + 1] = -y0 * sine + y1 * cosine;
          }
        }
      }
    }
    return inputGradient;
  }

  function repeatKvHeads(name, input, queryHeads) {
    const [batch, kvHeads, seq, headDim] = input.shape;
    const repeat = queryHeads / kvHeads;
    const output = zeros(name, [batch, queryHeads, seq, headDim]);
    for (let b = 0; b < batch; b += 1) {
      for (let h = 0; h < queryHeads; h += 1) {
        const sourceHead = Math.floor(h / repeat);
        for (let s = 0; s < seq; s += 1) {
          for (let k = 0; k < headDim; k += 1) {
            output.data[((b * queryHeads + h) * seq + s) * headDim + k] =
              input.data[((b * kvHeads + sourceHead) * seq + s) * headDim + k];
          }
        }
      }
    }
    return output;
  }

  function reduceKvHeads(name, repeatedGradient, kvHeads) {
    const [batch, queryHeads, seq, headDim] = repeatedGradient.shape;
    const repeat = queryHeads / kvHeads;
    const output = zeros(name, [batch, kvHeads, seq, headDim]);
    for (let b = 0; b < batch; b += 1) {
      for (let h = 0; h < queryHeads; h += 1) {
        const targetHead = Math.floor(h / repeat);
        for (let s = 0; s < seq; s += 1) {
          for (let k = 0; k < headDim; k += 1) {
            output.data[((b * kvHeads + targetHead) * seq + s) * headDim + k] +=
              repeatedGradient.data[((b * queryHeads + h) * seq + s) * headDim + k];
          }
        }
      }
    }
    return output;
  }

  function attentionForward(query, key, value) {
    const [batch, heads, seq, headDim] = query.shape;
    const scale = 1 / Math.sqrt(headDim);
    const scores = zeros("causal_scores", [batch, heads, seq, seq]);
    const weights = zeros("attention_weights", [batch, heads, seq, seq]);
    const context = zeros("context_heads", [batch, heads, seq, headDim]);
    for (let b = 0; b < batch; b += 1) {
      for (let h = 0; h < heads; h += 1) {
        for (let i = 0; i < seq; i += 1) {
          let rowMaximum = -Infinity;
          for (let j = 0; j < seq; j += 1) {
            const scoreIndex = ((b * heads + h) * seq + i) * seq + j;
            if (j > i) {
              scores.data[scoreIndex] = -Infinity;
              continue;
            }
            let dot = 0;
            const queryBase = ((b * heads + h) * seq + i) * headDim;
            const keyBase = ((b * heads + h) * seq + j) * headDim;
            for (let k = 0; k < headDim; k += 1) dot += query.data[queryBase + k] * key.data[keyBase + k];
            const score = dot * scale;
            scores.data[scoreIndex] = score;
            rowMaximum = Math.max(rowMaximum, score);
          }
          let denominator = 0;
          for (let j = 0; j <= i; j += 1) {
            const index = ((b * heads + h) * seq + i) * seq + j;
            const exponential = Math.exp(scores.data[index] - rowMaximum);
            weights.data[index] = exponential;
            denominator += exponential;
          }
          for (let j = 0; j <= i; j += 1) {
            const weightIndex = ((b * heads + h) * seq + i) * seq + j;
            weights.data[weightIndex] /= denominator;
            const weight = weights.data[weightIndex];
            const valueBase = ((b * heads + h) * seq + j) * headDim;
            const contextBase = ((b * heads + h) * seq + i) * headDim;
            for (let k = 0; k < headDim; k += 1) context.data[contextBase + k] += weight * value.data[valueBase + k];
          }
        }
      }
    }
    return { scores, weights, context };
  }

  function attentionBackward(query, key, value, weights, contextGradient) {
    const [batch, heads, seq, headDim] = query.shape;
    const scale = 1 / Math.sqrt(headDim);
    const weightsGradient = zeros("∂L/∂attention_weights", weights.shape);
    const scoresGradient = zeros("∂L/∂causal_scores", weights.shape);
    const queryGradient = zeros("∂L/∂q_rope", query.shape);
    const keyGradient = zeros("∂L/∂k_repeated", key.shape);
    const valueGradient = zeros("∂L/∂v_repeated", value.shape);

    for (let b = 0; b < batch; b += 1) {
      for (let h = 0; h < heads; h += 1) {
        for (let i = 0; i < seq; i += 1) {
          const contextBase = ((b * heads + h) * seq + i) * headDim;
          let softmaxDot = 0;
          for (let j = 0; j <= i; j += 1) {
            const matrixIndex = ((b * heads + h) * seq + i) * seq + j;
            const valueBase = ((b * heads + h) * seq + j) * headDim;
            let weightGradient = 0;
            for (let k = 0; k < headDim; k += 1) {
              weightGradient += contextGradient.data[contextBase + k] * value.data[valueBase + k];
              valueGradient.data[valueBase + k] += weights.data[matrixIndex] * contextGradient.data[contextBase + k];
            }
            weightsGradient.data[matrixIndex] = weightGradient;
            softmaxDot += weightGradient * weights.data[matrixIndex];
          }
          const queryBase = ((b * heads + h) * seq + i) * headDim;
          for (let j = 0; j <= i; j += 1) {
            const matrixIndex = ((b * heads + h) * seq + i) * seq + j;
            const scoreGradient = weights.data[matrixIndex] * (weightsGradient.data[matrixIndex] - softmaxDot);
            scoresGradient.data[matrixIndex] = scoreGradient;
            const keyBase = ((b * heads + h) * seq + j) * headDim;
            for (let k = 0; k < headDim; k += 1) {
              queryGradient.data[queryBase + k] += scoreGradient * key.data[keyBase + k] * scale;
              keyGradient.data[keyBase + k] += scoreGradient * query.data[queryBase + k] * scale;
            }
          }
        }
      }
    }
    return { weightsGradient, scoresGradient, queryGradient, keyGradient, valueGradient };
  }

  function siluForward(input) {
    const output = zeros("silu_gate", input.shape);
    const sigmoid = new Float64Array(input.data.length);
    for (let index = 0; index < input.data.length; index += 1) {
      const probability = 1 / (1 + Math.exp(-input.data[index]));
      sigmoid[index] = probability;
      output.data[index] = input.data[index] * probability;
    }
    return { output, sigmoid };
  }

  function crossEntropyForward(logits, targets) {
    const vocab = logits.shape[logits.shape.length - 1];
    const rows = logits.data.length / vocab;
    const probabilities = zeros("probabilities", logits.shape);
    let loss = 0;
    for (let row = 0; row < rows; row += 1) {
      const base = row * vocab;
      let maximum = -Infinity;
      for (let v = 0; v < vocab; v += 1) maximum = Math.max(maximum, logits.data[base + v]);
      let denominator = 0;
      for (let v = 0; v < vocab; v += 1) {
        const exponential = Math.exp(logits.data[base + v] - maximum);
        probabilities.data[base + v] = exponential;
        denominator += exponential;
      }
      for (let v = 0; v < vocab; v += 1) probabilities.data[base + v] /= denominator;
      const target = targets.data[row];
      loss -= Math.log(Math.max(probabilities.data[base + target], 1e-300));
    }
    return { probabilities, loss: loss / rows };
  }

  function crossEntropyBackward(probabilities, targets) {
    const vocab = probabilities.shape[probabilities.shape.length - 1];
    const rows = probabilities.data.length / vocab;
    const gradient = cloneTensor(probabilities, "∂L/∂logits");
    for (let row = 0; row < rows; row += 1) {
      gradient.data[row * vocab + targets.data[row]] -= 1;
    }
    for (let index = 0; index < gradient.data.length; index += 1) gradient.data[index] /= rows;
    return gradient;
  }

  function tensorStats(tensor) {
    let minimum = Infinity;
    let maximum = -Infinity;
    let sum = 0;
    let sumSquares = 0;
    let finite = 0;
    for (const value of tensor.data) {
      if (!Number.isFinite(value)) continue;
      minimum = Math.min(minimum, value);
      maximum = Math.max(maximum, value);
      sum += value;
      sumSquares += value * value;
      finite += 1;
    }
    return {
      min: finite ? minimum : NaN,
      max: finite ? maximum : NaN,
      mean: finite ? sum / finite : NaN,
      rms: finite ? Math.sqrt(sumSquares / finite) : NaN,
      finite,
      total: tensor.data.length,
    };
  }

  function run(rawConfig = PRESETS.standard, seed = 42, suppliedParameters = null) {
    const config = validateConfig({ ...rawConfig });
    const parameters = suppliedParameters ? cloneParameters(suppliedParameters) : initializeParameters(config, seed);
    const { batch: B, seq: S, hidden: d, heads: h, kvHeads: hk, intermediate: ff } = config;
    const headDim = d / h;
    const { tokens, targets } = makeTokens(config);
    const activations = { token_ids: tokens, target_ids: targets };

    activations.embedding = embeddingForward(tokens, parameters.embedding_table, d);
    const rms1Cache = rmsNormForward("rms1", activations.embedding, parameters.rms1_gamma, config.epsilon);
    activations.rms1 = rms1Cache.output;
    activations.q_linear = linearForward("q_linear", activations.rms1, parameters.W_q);
    activations.k_linear = linearForward("k_linear", activations.rms1, parameters.W_k);
    activations.v_linear = linearForward("v_linear", activations.rms1, parameters.W_v);
    activations.q_heads = splitHeads("q_heads", activations.q_linear, h, headDim);
    activations.k_heads = splitHeads("k_heads", activations.k_linear, hk, headDim);
    activations.v_heads = splitHeads("v_heads", activations.v_linear, hk, headDim);
    activations.q_rope = ropeForward("q_rope", activations.q_heads, config.ropeTheta);
    activations.k_rope = ropeForward("k_rope", activations.k_heads, config.ropeTheta);
    activations.k_repeated = repeatKvHeads("k_repeated", activations.k_rope, h);
    activations.v_repeated = repeatKvHeads("v_repeated", activations.v_heads, h);
    const attentionCache = attentionForward(activations.q_rope, activations.k_repeated, activations.v_repeated);
    activations.causal_scores = attentionCache.scores;
    activations.attention_weights = attentionCache.weights;
    activations.context_heads = attentionCache.context;
    activations.context = mergeHeads("context", activations.context_heads);
    activations.attn_out = linearForward("attn_out", activations.context, parameters.W_o);
    activations.residual1 = addTensors("residual1", activations.embedding, activations.attn_out);
    const rms2Cache = rmsNormForward("rms2", activations.residual1, parameters.rms2_gamma, config.epsilon);
    activations.rms2 = rms2Cache.output;
    activations.gate = linearForward("gate", activations.rms2, parameters.W_gate);
    activations.up = linearForward("up", activations.rms2, parameters.W_up);
    const siluCache = siluForward(activations.gate);
    activations.silu_gate = siluCache.output;
    activations.gated = multiplyTensors("gated", activations.silu_gate, activations.up);
    activations.ffn_out = linearForward("ffn_out", activations.gated, parameters.W_down);
    activations.residual2 = addTensors("residual2", activations.residual1, activations.ffn_out);
    const finalRmsCache = rmsNormForward("final_norm", activations.residual2, parameters.final_rms_gamma, config.epsilon);
    activations.final_norm = finalRmsCache.output;
    activations.logits = linearForward("logits", activations.final_norm, parameters.W_lm_head);
    const lossCache = crossEntropyForward(activations.logits, targets);
    activations.probabilities = lossCache.probabilities;
    activations.loss = makeTensor("cross_entropy_loss", [1], [lossCache.loss]);

    const gradients = { activations: {}, parameters: {} };
    const ga = gradients.activations;
    const gp = gradients.parameters;
    ga.logits = crossEntropyBackward(activations.probabilities, targets);
    const lmBackward = linearBackward(
      activations.final_norm, parameters.W_lm_head, ga.logits,
      "∂L/∂final_norm", "∂L/∂W_lm_head",
    );
    ga.final_norm = lmBackward.inputGradient;
    gp.W_lm_head = lmBackward.weightGradient;
    const finalRmsBackward = rmsNormBackward(
      activations.residual2, parameters.final_rms_gamma, finalRmsCache.inverseRms, ga.final_norm,
      "∂L/∂residual2", "∂L/∂final_rms_gamma",
    );
    ga.residual2 = finalRmsBackward.inputGradient;
    gp.final_rms_gamma = finalRmsBackward.gammaGradient;

    ga.residual1_direct = cloneTensor(ga.residual2, "∂L/∂residual1 (residual branch)");
    ga.ffn_out = cloneTensor(ga.residual2, "∂L/∂ffn_out");
    const downBackward = linearBackward(
      activations.gated, parameters.W_down, ga.ffn_out,
      "∂L/∂gated", "∂L/∂W_down",
    );
    ga.gated = downBackward.inputGradient;
    gp.W_down = downBackward.weightGradient;
    ga.silu_gate = zeros("∂L/∂silu_gate", activations.silu_gate.shape);
    ga.up = zeros("∂L/∂up", activations.up.shape);
    for (let index = 0; index < ga.gated.data.length; index += 1) {
      ga.silu_gate.data[index] = ga.gated.data[index] * activations.up.data[index];
      ga.up.data[index] = ga.gated.data[index] * activations.silu_gate.data[index];
    }
    ga.gate = zeros("∂L/∂gate", activations.gate.shape);
    for (let index = 0; index < ga.gate.data.length; index += 1) {
      const gateValue = activations.gate.data[index];
      const sigmoid = siluCache.sigmoid[index];
      const derivative = sigmoid + gateValue * sigmoid * (1 - sigmoid);
      ga.gate.data[index] = ga.silu_gate.data[index] * derivative;
    }
    const gateBackward = linearBackward(
      activations.rms2, parameters.W_gate, ga.gate,
      "∂L/∂rms2 from gate", "∂L/∂W_gate",
    );
    const upBackward = linearBackward(
      activations.rms2, parameters.W_up, ga.up,
      "∂L/∂rms2 from up", "∂L/∂W_up",
    );
    gp.W_gate = gateBackward.weightGradient;
    gp.W_up = upBackward.weightGradient;
    ga.rms2_gate = gateBackward.inputGradient;
    ga.rms2_up = upBackward.inputGradient;
    ga.rms2 = addTensors("∂L/∂rms2", ga.rms2_gate, ga.rms2_up);
    const rms2Backward = rmsNormBackward(
      activations.residual1, parameters.rms2_gamma, rms2Cache.inverseRms, ga.rms2,
      "∂L/∂residual1 (FFN branch)", "∂L/∂rms2_gamma",
    );
    ga.residual1_ffn = rms2Backward.inputGradient;
    gp.rms2_gamma = rms2Backward.gammaGradient;
    ga.residual1 = addTensors("∂L/∂residual1 (accumulated)", ga.residual1_direct, ga.residual1_ffn);

    ga.embedding_direct = cloneTensor(ga.residual1, "∂L/∂embedding (residual branch)");
    ga.attn_out = cloneTensor(ga.residual1, "∂L/∂attn_out");
    const outputBackward = linearBackward(
      activations.context, parameters.W_o, ga.attn_out,
      "∂L/∂context", "∂L/∂W_o",
    );
    ga.context = outputBackward.inputGradient;
    gp.W_o = outputBackward.weightGradient;
    ga.context_heads = splitHeads("∂L/∂context_heads", ga.context, h, headDim);
    const attentionBackwardCache = attentionBackward(
      activations.q_rope, activations.k_repeated, activations.v_repeated,
      activations.attention_weights, ga.context_heads,
    );
    ga.attention_weights = attentionBackwardCache.weightsGradient;
    ga.causal_scores = attentionBackwardCache.scoresGradient;
    ga.q_rope = attentionBackwardCache.queryGradient;
    ga.k_repeated = attentionBackwardCache.keyGradient;
    ga.v_repeated = attentionBackwardCache.valueGradient;
    ga.k_rope = reduceKvHeads("∂L/∂k_rope", ga.k_repeated, hk);
    ga.v_heads = reduceKvHeads("∂L/∂v_heads", ga.v_repeated, hk);
    ga.q_heads = ropeBackward("∂L/∂q_heads", ga.q_rope, config.ropeTheta);
    ga.k_heads = ropeBackward("∂L/∂k_heads", ga.k_rope, config.ropeTheta);
    ga.q_linear = mergeHeads("∂L/∂q_linear", ga.q_heads);
    ga.k_linear = mergeHeads("∂L/∂k_linear", ga.k_heads);
    ga.v_linear = mergeHeads("∂L/∂v_linear", ga.v_heads);
    const qBackward = linearBackward(
      activations.rms1, parameters.W_q, ga.q_linear,
      "∂L/∂rms1 from Q", "∂L/∂W_q",
    );
    const kBackward = linearBackward(
      activations.rms1, parameters.W_k, ga.k_linear,
      "∂L/∂rms1 from K", "∂L/∂W_k",
    );
    const vBackward = linearBackward(
      activations.rms1, parameters.W_v, ga.v_linear,
      "∂L/∂rms1 from V", "∂L/∂W_v",
    );
    gp.W_q = qBackward.weightGradient;
    gp.W_k = kBackward.weightGradient;
    gp.W_v = vBackward.weightGradient;
    ga.rms1_q = qBackward.inputGradient;
    ga.rms1_k = kBackward.inputGradient;
    ga.rms1_v = vBackward.inputGradient;
    ga.rms1 = addTensors("∂L/∂rms1", addTensors("q+k", ga.rms1_q, ga.rms1_k), ga.rms1_v);
    const rms1Backward = rmsNormBackward(
      activations.embedding, parameters.rms1_gamma, rms1Cache.inverseRms, ga.rms1,
      "∂L/∂embedding (attention branch)", "∂L/∂rms1_gamma",
    );
    ga.embedding_attention = rms1Backward.inputGradient;
    gp.rms1_gamma = rms1Backward.gammaGradient;
    ga.embedding = addTensors("∂L/∂embedding (accumulated)", ga.embedding_direct, ga.embedding_attention);
    gp.embedding_table = embeddingBackward(tokens, ga.embedding, parameters.embedding_table.shape);

    const parameterCount = Object.values(parameters).reduce((sum, tensor) => sum + tensor.data.length, 0);
    return {
      config, seed: Number(seed), parameters, activations, gradients,
      loss: lossCache.loss, parameterCount, headDim,
      vocabulary: makeVocabulary(config.vocab),
    };
  }

  function makeVocabulary(size) {
    const base = [
      "<pad>", "<bos>", "分布式", "AI", "系统", "把", "token", "变成", "向量", "并",
      "通过", "attention", "聚合", "上下文", "再", "经过", "SwiGLU", "加回", "残差", "流",
      "loss", "产生", "梯度", "反向", "穿过", "矩阵", "更新", "权重", "完成", "一步",
      "训练", "<eos>",
    ];
    const vocabulary = [];
    for (let index = 0; index < size; index += 1) vocabulary.push(base[index] || `词元_${index}`);
    return vocabulary;
  }

  function sgdStep(result, learningRate = 0.05) {
    if (!(learningRate > 0) || !Number.isFinite(learningRate)) throw new Error("learningRate must be positive");
    const updated = cloneParameters(result.parameters);
    for (const [name, tensor] of Object.entries(updated)) {
      const gradient = result.gradients.parameters[name];
      if (!gradient) throw new Error(`Missing gradient for ${name}`);
      for (let index = 0; index < tensor.data.length; index += 1) {
        tensor.data[index] -= learningRate * gradient.data[index];
      }
    }
    const next = run(result.config, result.seed, updated);
    return { before: result, after: next, learningRate, delta: next.loss - result.loss };
  }

  return {
    PRESETS,
    makeTensor,
    cloneTensor,
    cloneParameters,
    tensorStats,
    run,
    sgdStep,
  };
});
