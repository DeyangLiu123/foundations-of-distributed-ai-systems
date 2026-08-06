const assert = require("node:assert/strict");
const { test } = require("node:test");

const Engine = require("../M2 Transformer与大模型/实践代码/L12_tensor_engine.js");
const Registry = require("../M2 Transformer与大模型/实践代码/L12_stream_registry.js");

test("global stream registry covers every forward tensor exactly once", () => {
  const result = Engine.run(Engine.PRESETS.standard, 42);
  const report = Registry.coverage("forward", result);
  assert.deepEqual(report, { missing: [], duplicates: [], unexpected: [], complete: true });
});

test("global stream registry covers every backward tensor exactly once", () => {
  const result = Engine.run(Engine.PRESETS.standard, 42);
  const report = Registry.coverage("backward", result);
  assert.deepEqual(report, { missing: [], duplicates: [], unexpected: [], complete: true });
});

test("all registry entries point to tensors with non-empty shapes", () => {
  const result = Engine.run(Engine.PRESETS.compact, 42);
  for (const mode of ["forward", "backward"]) {
    const sources = mode === "forward"
      ? { a: result.activations, p: result.parameters }
      : { ga: result.gradients.activations, gp: result.gradients.parameters };
    for (const entry of Registry.flatten(mode)) {
      assert.ok(sources[entry.source][entry.key].shape.length > 0, `${mode}:${entry.key}`);
    }
  }
});

test("forward and backward execution graphs have no dangling or duplicated tensor nodes", () => {
  for (const mode of ["forward", "backward"]) {
    const report = Registry.graphIntegrity(mode);
    assert.deepEqual(report, {
      duplicateNodeIds: [],
      danglingEdges: [],
      missingTensors: [],
      unexpectedTensors: [],
      duplicateTensors: [],
      complete: true,
    });
  }
});

test("execution graph edges move downstream and therefore remain acyclic", () => {
  for (const mode of ["forward", "backward"]) {
    const graph = Registry.graphs[mode];
    const nodes = new Map(graph.nodes.map((node) => [node.id, node]));
    for (const connection of graph.edges) {
      assert.ok(nodes.get(connection.from).x < nodes.get(connection.to).x, `${mode}: ${connection.from} must precede ${connection.to}`);
    }
  }
});
