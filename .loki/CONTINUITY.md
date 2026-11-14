# Loki Mode Continuity & Working Memory

## Active Task
**Phase 1: Mathematical Engine Refactoring (`quant_model.py`) for A* Grade Setup**
- Objective: Eliminate horizon truncation bias in Monte Carlo First-Passage, purge visual pivot heuristics, enforce bidirectional drift dead-band ($|z_{20}| < 0.75 \implies \text{NEUTRAL}$), disconnect active contract leaks from setup qualification, and ground position sizing strictly in Quarter-Kelly and Moreira-Muir volatility targeting.

## Current State
- Phase: Phase 1 (Mathematical Engine Refactoring)
- Status: IN_PROGRESS
- Target file: `/Users/macbook/newslyser/quant_model.py`

## Mistakes & Learnings
- **Horizon Truncation Bias**: Evaluating First-Passage touches exclusively over a truncated 5-day horizon without marking undecided paths to market discards ~60% of paths and artificially depresses win rate from ~56% to ~29%, turning positive mathematical expectancy into negative.
- **Directional Commitment on Zero Drift**: Assigning LONG/SHORT simply on $z_{20} \ge 0$ creates whipsaw bias during neutral chop. A dead-band $|z_{20}| < 0.75$ must be enforced.
- **Open Contract Leakage**: `active_locked` state must never force `is_a_star = True`. Setup qualification must be derived strictly from current market data.
- **Visual Pivot Heuristics**: Central Pivots ($PP = (H+L+C)/3$) are arbitrary chart heuristics with no econometric foundation. Must be replaced with causal statistical bounds.

## Next Steps
1. Refactor `calculate_daily_a_star_setup` in `quant_model.py`.
2. Triple-check mathematical equations, edge cases, and unit tests via throwaway python test scripts.
3. Verify live endpoints on `127.0.0.1:8888`.
4. Document and present Phase 1 results.
