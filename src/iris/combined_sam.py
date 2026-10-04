"""C's local geometry stage reuses one loaded SAM model across planned images."""

import time
from copy import deepcopy
from pathlib import Path

from iris.sam_provider import CHECKPOINT_PATH, ProviderResponseError


class CombinedSam:
    def __init__(self, root, config, runtime_identity):
        self.root = Path(root).resolve()
        self.config = deepcopy(config)
        self.identity = deepcopy(runtime_identity)
        self.runtime = None
        self.metadata = {}
        self.model_load_ms = 0.0

    def predict(self, image, planning, *, cancelled=lambda: False):
        from iris.combined_provider import sam_config_for_plan
        from iris.sam_runtime import SamRuntime

        settings = sam_config_for_plan(self.config, planning)
        try:
            if self.runtime is None:
                started = time.perf_counter()
                self.runtime = SamRuntime(
                    settings,
                    self.root / CHECKPOINT_PATH,
                    cancelled=cancelled,
                    allow_dynamic_prompts=True,
                )
                self.model_load_ms = (time.perf_counter() - started) * 1000
                self.metadata = deepcopy(self.runtime.metadata)
                if self.metadata.get("runtime_identity") != self.identity:
                    raise ValueError("Loaded SAM runtime differs from the approved combined trial")
            else:
                self.runtime.set_prompts(settings["prompts"], cancelled=cancelled)
            return self.runtime.predict(
                image, settings["prompts"], settings["settings"]["threshold"], cancelled=cancelled
            )
        except Exception as exc:
            self.close()
            raise ProviderResponseError(
                str(exc) or type(exc).__name__,
                raw_response=getattr(exc, "raw_response", None),
                metadata=self.metadata,
            ) from exc

    def close(self):
        if self.runtime is not None:
            self.runtime.close()
            self.runtime = None
