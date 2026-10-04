"use strict";

((root, factory) => {
  const tools = factory();
  if (typeof module === "object" && module.exports) module.exports = tools;
  if (typeof window !== "undefined") root.IRISTaxonomyTools = tools;
})(typeof globalThis !== "undefined" ? globalThis : this, () => {
  const builtinId = "iris-objects-v1";
  const builtin = {
    id: builtinId,
    classes: [
      { id: "person", name: "Person", coco_id: 1,
        definition: "A visible human, including a rider. Enclose the visible extent of each person; do not infer a box for a fully occluded person." },
      { id: "car", name: "Car", coco_id: 3,
        definition: "A passenger car, including an SUV or passenger minivan. Exclude buses, trucks, motorcycles and bicycles. Enclose the visible extent of each car." },
    ],
  };
  const snapshot = (value) => structuredClone(value || builtin);
  const versionLabel = (value) => value?.id === builtinId
    ? "Original person / car classes"
    : `Class version ${value?.version ?? "saved"}`;
  const className = (taxonomy, id) => taxonomy?.classes.find((item) => item.id === id)?.name || id;
  const hasClass = (taxonomy, id) => Boolean(taxonomy?.classes.some((item) => item.id === id));
  function mappingComplete(categories, mapping, taxonomy) {
    return categories.every((item) => mapping[String(item.id)] === "exclude" || hasClass(taxonomy, mapping[String(item.id)]));
  }
  function editableClasses(taxonomy) {
    return snapshot(taxonomy).classes.map((item) => ({ ...item, coco_id: item.coco_id ?? null }));
  }
  function classColor(taxonomy, id) {
    const index = taxonomy.classes.findIndex((item) => item.id === id);
    return index < 0 ? "#e0e3e5" : ["#8ef1ac", "#83caff", "#ffd17d", "#e0a4fa", "#7ce3df", "#ffadaf"][index % 6];
  }
  return { builtinId, snapshot, versionLabel, className, hasClass, mappingComplete, editableClasses, classColor };
});
