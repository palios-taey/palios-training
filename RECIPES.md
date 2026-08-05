# Recipes (public)

Public launch recipes live under `dense-9b/recipes/` and `careers-qwen/` shell entry points
that do not embed private training material.

Private bake architecture notes, standards maps, substrate physics write-ups, and module
intake specs are **not** in this tree. Resolve them only via:

```bash
: "${GOVERNED_SFT_ROOT:?set GOVERNED_SFT_ROOT}"
# private snapshots (evidence only): $GOVERNED_SFT_ROOT/sources/
# standards: $GOVERNED_SFT_ROOT/standards/
```

Do not paste private recipe prose into this file.
