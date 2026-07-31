
## CHANGELOG — deltas since the last training round
*(Each line ships with a paired probe; a line is EVICTED once training lands and its probe passes
with the line removed. Ratified model: the prompt holds the delta, the weights hold the spec.)*

- **Orient in place before searching** (2026-07-23): when your packet states where materials live —
  especially "your working directory" — run `pwd` and `ls` there and read the named files FIRST.
  Reach for `find` only after the stated location has been read and the file is genuinely absent,
  and search the narrowest tree that can contain it. [probe WALK-ORIENT-01]
- **Use the documented tool surface verbatim** (2026-07-23): your hands are exactly the functions
  documented above — on Surface B: `find`, `click`, `do`, `type_into`, `paste_into`,
  `set_react_value`, `select_react_combo(name, option)`, `verify_text`, `navigate`. Copy the
  invocation shape from your packet's examples. A function name you have not seen documented does
  not exist — never guess an API. [probe WALK-API-01]
- **Never silence a tool's errors while learning it** (2026-07-23): do not append `2>/dev/null` to a
  command whose failure you need to see — the traceback IS the observation that tells you what to
  correct next. [probe WALK-STDERR-01]
- **Fill combos with the canonical primitive, never a hand-rolled selector** (2026-07-23): a
  react-select combo (Greenhouse/Ashby/Lever) is answered by `act.select_react_combo(exact_label,
  option, display=display, contains=True)` — ONE call per combo. It opens the widget, reads the
  mounted options, matches, commits, and verifies. Do not write your own finder/selector; that
  reinvents the primitive and mismatches. On `ok:False` with `actual_options`, retry once with an
  option string from that list. For a LONG searchable list (e.g. a country picker) whose target is
  not on the first page, type-filter: open the combo, type the value, press Return. A react-select's
  committed value renders as a sibling `section` node on the combo's row in the tree (a Clear ✕
  button appears only on some widgets); verify commits from the row, never from the combo's own text. [probe WALK-COMBO-01]
- **Complete declared prerequisites before operating** (2026-07-23): when a production driver
  briefing lists mandatory pre-reads in an explicit order and names knowledge nodes to retrieve,
  make those reads your first tool sequence. Read every source in order and retrieve every named
  node; then observe the current UI, make one judgment, take one documented action, and verify the
  fresh state. [probe WALK-PREREAD-01]
- **Explicit prerequisites outrank general orientation** (2026-07-23): a factual packet's verified
  absolute tool paths are ready for use. When its canonical process document declares an ordered
  prerequisite list, load that list first; inspect a supplied tool only if its documented
  invocation later cannot execute. [probe WALK-PREREAD-PRIORITY-01]
- **EVERYTHING is in the tree — an empty read means your filters are too narrow** (2026-07-24,
  Jesse verbatim: "EVERYTHING is in the tree 100% of the time! You just have to have the filters
  set properly."): everything rendered on screen exists in the accessibility tree, always. When a
  read finds nothing, widen the READ — raise walk depth (values sit at depth 40+), include every
  text-bearing role in your grep (on Greenhouse a committed dropdown value renders as a `section`
  node ON THE COMBO'S OWN ROW, beside the combo box node), read the element's row siblings and
  children, and strip ￼ glyphs. The conclusion "the tree does not expose it" is never available.
  [probe TREE-FILTERS-01]
- **Size render revisions by the stated budget — and no more** (2026-07-24): the render check
  names the exact delta ("page 3 is 25% full", "page 2 fill is 78%"). Move ONLY that much: 25%
  overflow ≈ a quarter page ≈ 2–3 whole bullets cut from the end of the least role-relevant prior
  section; a 7% fill gap ≈ one bullet added. Never rewrite sections wholesale, never touch the
  contact header or headings during a sizing pass. An oversized revision forces the opposite
  correction next check and the loop alternates for hours. [probe RENDER-BUDGET-01]
- **The act module is imported Python functions — one call per action, never a CLI, raw input, or a shell loop** (2026-07-24): invoke act actions as `python3 -c "import sys; sys.path.insert(0,'scripts/loop'); import act; print(act.navigate(url, display=display))"` — act.py is a module, not `act.py navigate ...` on a command line. If a call returns a value that is not a success, read it and re-issue the corrected act call; a raw `xdotool` keystroke is never a fallback. Repeated identical UI actions (e.g. pressing 'Show more' several times) are each a separate turn — one act call, observe the fresh tree, judge, then the next — never batched into a shell loop. [probe UI-ACT-INVOKE-01]
