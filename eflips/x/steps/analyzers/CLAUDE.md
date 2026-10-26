# Analyzers

- Analyzers must not modify the database.
- `visualize()` is optional for table-only analyzers; the user views or exports tables themselves.
- Generic analyzers (`input_analyzers.py`, `output_analyzers.py`) return data only and produce undecorated, standard
  matplotlib/plotly figures from `visualize()`, structured so a subclass can pre-/post-process them.
- BVG-specific styling belongs in subclasses in `bvg_tools.py`, not in the generic analyzers.
- Matplotlib: `layout="constrained"`, never `tight_layout`.
