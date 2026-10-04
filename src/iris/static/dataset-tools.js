"use strict";

((root, factory) => {
  const tools = factory();
  if (typeof module === "object" && module.exports) module.exports = tools;
  if (typeof window !== "undefined") root.IRISDatasetTools = tools;
})(typeof globalThis !== "undefined" ? globalThis : this, () => {
  const builtinId = "iris-objects-v1";
  const taxonomyId = (record) => record?.taxonomy_id || record?.taxonomy?.id || record?.manifest?.taxonomy?.id || builtinId;
  const mlSupported = (record) => Boolean(record) &&
    (typeof record.ml_supported === "boolean" ? record.ml_supported : taxonomyId(record) === builtinId);
  const taxonomyOf = (record) => record?.taxonomy?.classes ? record.taxonomy
    : record?.config?.taxonomy?.classes ? record.config.taxonomy
      : record?.manifest?.taxonomy?.classes ? record.manifest.taxonomy
        : { id: builtinId, classes: [{ id: "person", name: "Person", coco_id: 1 }, { id: "car", name: "Car", coco_id: 3 }] };
  const canonical = (value) => JSON.stringify(value, function (key, item) {
    if (item && typeof item === "object" && !Array.isArray(item))
      return Object.fromEntries(Object.keys(item).sort().map((name) => [name, item[name]]));
    return item;
  });
  function modelCompatibility(dataset, model, purpose = "evaluation") {
    if (!dataset) return { compatible: false, reason: "Choose a frozen dataset release first." };
    if (!model) return { compatible: false, reason: "Choose a model." };
    if (!mlSupported(dataset)) return { compatible: false, reason: dataset.ml_limitation || "This dataset is not supported by the current runtime." };
    if (purpose === "training" && !model.training)
      return { compatible: false, reason: "This model cannot be trained by the current runtime." };
    const taxonomy = taxonomyOf(dataset);
    if (model.origin === "trained") {
      const sameId = taxonomyId(model) === taxonomy.id;
      const sameSnapshot = model.taxonomy?.classes && canonical(model.taxonomy) === canonical(taxonomy);
      const legacy = taxonomy.id === builtinId && !model.taxonomy;
      const sameInternal = !model.class_mapping || !dataset.class_mapping || canonical(model.class_mapping) === canonical(dataset.class_mapping);
      const sameOutputs = !model.output_class_mapping || !dataset.coco_mapping || canonical(model.output_class_mapping) === canonical(dataset.coco_mapping);
      return sameId && (sameSnapshot || legacy) && sameInternal && sameOutputs
        ? { compatible: true, reason: "Same saved class definitions as this dataset." }
        : { compatible: false, reason: "This checkpoint uses different saved class definitions. Choose a checkpoint from the same class version." };
    }
    if (purpose === "training") return { compatible: true, reason: "Adapt the official checkpoint to this dataset's classes; unmapped classes start with new prediction weights." };
    const unmapped = taxonomy.classes.filter((item) => !Number.isInteger(item.coco_id) ||
      model.classes?.length && !model.classes.some((category) => category.id === item.coco_id));
    return unmapped.length
      ? { compatible: false, reason: `Official model cannot evaluate unmapped classes: ${unmapped.map((item) => item.name).join(", ")}. Train a matching checkpoint first.` }
      : { compatible: true, reason: "Every dataset class has an explicit mapping to this official detector." };
  }
  const className = (taxonomy, id) => taxonomy.classes.find((item) => item.id === id)?.name || id;
  const aggregateFilter = (analysis) => analysis?.aggregate_filter || "all";
  function analysisScopes(analysis, taxonomy) {
    return [{ value: aggregateFilter(analysis), label: "All classes" },
      ...taxonomy.classes.filter((item) => Object.hasOwn(analysis.summary, item.id))
        .map((item) => ({ value: item.id, label: item.name }))];
  }
  function displayedDetections(taxonomy, detections, threshold) {
    return (detections || []).map((item, index) => ({ ...item, index })).filter((item) =>
      !item.ignored && Number.isFinite(item.score) && item.score >= threshold &&
      taxonomy.classes.some((category) => category.id === item.label),
    );
  }
  const compatibleParents = (records, taxonomy, project) => records.filter((record) =>
    taxonomyId(record) === taxonomy && (!record.project_id || record.project_id === project),
  );

  function revisionTokens(groups) {
    const tokens = {};
    for (const group of groups) for (const frame of group.frames) {
      if (typeof frame.annotation_revision_id !== "string" || !frame.annotation_revision_id ||
          Object.hasOwn(tokens, frame.id))
        throw new Error("Refresh dataset candidates before freezing: the reviewed revision list is incomplete.");
      tokens[frame.id] = frame.annotation_revision_id;
    }
    return tokens;
  }

  function classRows(record, fallbackTaxonomy) {
    const taxonomy = record.taxonomy || record.manifest?.taxonomy || fallbackTaxonomy;
    const legacy = taxonomy.id === builtinId;
    const labels = record.class_mapping || record.manifest?.class_mapping || (legacy ? { person: 1, car: 2 } : {});
    const exports = record.coco_mapping || record.manifest?.coco_mapping || (legacy ? { person: 1, car: 3 } : {});
    return taxonomy.classes.map((category) => ({
      ...category,
      count: record.summary?.class_counts?.[category.id] || 0,
      class_id: labels[category.id] ?? null,
      export_id: exports[category.id] ?? null,
      source_coco_id: category.coco_id ?? null,
    }));
  }
  return { taxonomyId, taxonomyOf, mlSupported, modelCompatibility, className, aggregateFilter, analysisScopes, displayedDetections, compatibleParents, revisionTokens, classRows };
});
