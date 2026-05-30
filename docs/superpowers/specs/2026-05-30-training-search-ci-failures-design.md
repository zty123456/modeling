# Training Search CI Failure Fix Design

## Goal

Resolve the remaining `test_training_search_util.py` failures exposed by PR #193
without changing the intended training-search report semantics or the user's
local reproduction grid.

## Scope

The fix contains three narrowly scoped changes:

1. Normalize `vpp_chunks` to `1` when `pp_schedule` is neither `interleaved`
   nor `dualpipev`. These schedules do not consume virtual pipeline chunks.
2. Keep `format_results()` sorted by descending `tokens_per_sec`, which is the
   current report contract, and update the stale MFU-sorting regression test.
3. Preserve the existing export-time `quant_preset` propagation so regenerated
   best-config Excel files use the same model quantization preset as search.

The workflow configuration and the user's local `training_param_grid` values
are outside this fix.

## Implementation

- Update `_make_strategy_from_config()` in
  `python/zrt/training/search/training_search_util.py` to normalize
  `vpp_chunks`.
- Update the stale sorting test in
  `tests/training/test_training_search_util.py` to assert descending
  `tokens_per_sec`.
- Retain the existing local `quant_preset` propagation in
  `export_best_configs_excel()`.

## Verification

Run the three previously failing tests first, then the complete
`tests/training/test_training_search_util.py` file. Finally run `compileall`
and `git diff --check` for the touched files.

