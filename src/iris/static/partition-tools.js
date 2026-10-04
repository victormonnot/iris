"use strict";
((root, factory) => {
  const tools = factory();
  if (typeof module === "object" && module.exports) module.exports = tools;
  else root.IRISPartitionTools = tools;
})(typeof window === "undefined" ? globalThis : window, () => {
  function validatePlan(plan, candidates) {
    if (!plan || !candidates || plan.taxonomy_id !== candidates.taxonomy?.id)
      throw new Error("Class definitions changed. Preview a new partition plan.");
    if (!plan.can_freeze || plan.blockers?.length)
      throw new Error("Resolve the plan's blockers before applying partitions.");
    const groups = candidates.groups || [];
    const frames = groups.flatMap((group) => group.frames);
    const actual = new Map(frames.map((frame) => [frame.id, frame.annotation_revision_id]));
    const planned = plan.frame_ids || [];
    if (actual.size !== planned.length || new Set(planned).size !== planned.length ||
        planned.some((id) => !actual.has(id) || !actual.get(id) || actual.get(id) !== plan.expected_revisions?.[id]) ||
        Object.keys(plan.expected_revisions || {}).length !== actual.size)
      throw new Error("Eligible frames or reviewed revisions changed. Refresh candidates and preview a new plan.");
    const choices = new Map();
    if (Object.keys(plan.splits || {}).length !== groups.length)
      throw new Error("Scene groups changed. Preview a new partition plan.");
    for (const group of groups) {
      const split = plan.splits[group.scene_group];
      const reserved = new Set([group.reserved_split, ...group.frames.map((frame) => frame.reserved_split)].filter(Boolean));
      if (!["train", "val", "test"].includes(split) || reserved.size > 1 || (reserved.size === 1 && !reserved.has(split)))
        throw new Error("Split reservations changed. Refresh candidates and preview a new plan.");
      choices.set(group.scene_group, split);
    }
    return choices;
  }
  return { validatePlan };
});
