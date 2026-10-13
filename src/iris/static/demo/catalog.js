"use strict";

// Read-only empty catalogues for native workspaces without recorded evidence.
// A saved response takes precedence in the adapter; this module never runs work.
((root, factory) => {
  const tools = factory();
  if (typeof module === "object" && module.exports) module.exports = tools;
  else root.IRISDemoCatalog = tools;
})(typeof globalThis !== "undefined" ? globalThis : this, () => {
  const reason = "Unavailable in demo mode. Install IRIS locally to run this operation.";
  const unavailable = () => ({ status: "unavailable", available: false, demo: true, reason });

  function create(seed) {
    const taxonomies = structuredClone(seed.responses[`/api/projects/${seed.project_id}/taxonomies`]);
    const models = structuredClone(seed.responses["/api/models"] || []).map((model) => ({
      ...model, ...unavailable(), runtime_load_verified: false,
    }));
    const session = `/api/sessions/${seed.session_id}`;
    const emptyLists = new Set([
      "/api/datasets", "/api/trainings", "/api/evaluations", "/api/model-references",
      "/api/experiments", "/api/benchmarks", "/api/model-exports",
      "/api/temporal/sequences", "/api/temporal/datasets",
      "/api/temporal/tracking-studies", "/api/temporal/tracking-selections",
      "/api/temporal/pipeline-bundles",
      `${session}/preannotations`, `${session}/dinox-batches`, `${session}/assistance-batches`,
    ]);
    const candidates = new Set(["/api/dataset-candidates", "/api/benchmark-candidates"]);
    const runtime = () => ({ ...unavailable(), installation: reason, packages: {}, max_frames: 500 });

    function response(path, searchParams = new URLSearchParams()) {
      if (emptyLists.has(path)) return [];
      if (candidates.has(path)) {
        const id = searchParams.get("taxonomy_id") || taxonomies.current_taxonomy_id;
        const taxonomy = taxonomies.versions.find((item) => item.id === id);
        if (!taxonomy) throw Object.assign(new Error("This class version is not part of the demo."), { status: 404 });
        return structuredClone({
          taxonomy, taxonomies: taxonomies.versions, groups: [], excluded: {},
          demo: true, reason: "No frozen dataset or independent reference is included in this demo.",
        });
      }
      if (path === "/api/training/devices") return {
        devices: [
          { id: "cpu", label: "CPU", available: false, reason },
          { id: "cuda", label: "NVIDIA CUDA GPU", available: false, reason },
        ], notes: [reason], demo: true,
      };
      if (path === "/api/model-exports/candidates") return {
        models: [], ...unavailable(),
      };
      if (path === "/api/preannotation-providers") return {
        providers: [{
          id: "local_detector", label: "Recorded detectors", local: true,
          ...unavailable(), capabilities: { creates_boxes: false, execution: "unavailable" },
          models: structuredClone(models),
        }],
      };
      if (path === "/api/dinox/provider") return {
        ...unavailable(), id: "dinox", model: "DINO-X-1.0", key_source: null,
      };
      if (path === "/api/benchmark-providers") return {
        multimodal: { ...unavailable(), model: "gpt-6-astra" },
        // Public form choices mirror benchmark_multimodal/segmentation.py.
        // They describe local configuration, not a hosted provider or credential.
        multimodal_settings: {
          defaults: { image_long_edge: 1536, reasoning_effort: "low", max_output_tokens: 4096 },
          image_long_edges: [512, 1024, 1536, 2048],
          reasoning_efforts: ["low", "medium", "high", "xhigh", "max"],
          max_output_tokens: { min: 1024, max: 8192 }, detail: "original",
        },
        segmentation: {
          ...unavailable(), model_id: "sam3", devices: [{ id: "cuda", available: false, reason }],
        },
        segmentation_settings: {
          defaults: { threshold: 0.5, device: "cuda" }, threshold: { min: 0, max: 1 },
          prompt_limits: { min_length: 1, max_length: 120, max_tokens: 30, max_classes: 100 },
          devices: ["cuda"],
        },
        combined: { ...unavailable(), model_id: "gpt-6-astra+sam3" },
      };
      if (path === "/api/temporal/tracking-status") return runtime();
      if (path === "/api/temporal/tracking-study-sources") return { datasets: [], sequences: [] };
      if (path === "/api/temporal/tracking-study-status") return { ...unavailable(), runtime: runtime() };
      if (["/api/temporal/tracking-selection-sources", "/api/temporal/pipeline-bundle-sources"].includes(path))
        return { sources: [] };
      if (["/api/temporal/tracking-selection-status", "/api/temporal/pipeline-bundle-status"].includes(path))
        return unavailable();
      if (path === "/api/temporal/tracking-quality-status") return {
        ...unavailable(),
        protocol: { name: "iris-observed-identity-v1" },
        limits: {
          max_frames: 500, max_frame_objects: 512, max_identity_count: 512,
          max_assignment_work: 25000000,
        },
      };
      return undefined;
    }

    return {
      queryParameters: (path) => candidates.has(path) ? ["taxonomy_id"] : [],
      response,
    };
  }

  return { create };
});
