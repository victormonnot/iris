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
  return { taxonomyId, mlSupported, compatibleParents, revisionTokens, classRows };
});
