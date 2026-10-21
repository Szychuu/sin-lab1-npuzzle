# ===== CELL 1: proposed heuristics (paste after `from search import *`) =====
# If your Euclid / RowCol from tasks 2-3 already exist under other names,
# you can replace euclid_tiles / rowcol_tiles below with your own functions.
import math, io, contextlib, statistics, random

def _pos(state, tile, size):
    i = state.index(tile)
    return i // size, i % size

def euclid_tiles(state, end_state):
    size = int(math.sqrt(len(state)))
    tot = 0.0
    for t in range(1, len(state)):          # tiles 1..n^2-1, blank (0) skipped
        r, c = _pos(state, t, size)
        gr, gc = _pos(end_state, t, size)
        tot += math.hypot(r - gr, c - gc)
    return tot

def rowcol_tiles(state, end_state):
    size = int(math.sqrt(len(state)))
    tot = 0
    for t in range(1, len(state)):
        r, c = _pos(state, t, size)
        gr, gc = _pos(end_state, t, size)
        tot += (r != gr) + (c != gc)
    return tot

# (a) OPTIMAL: max(E, RC) + LC
# E <= MD and RC <= MD  =>  max(E, RC) + LC <= MD + LC <= h*  (admissible)
def proposed_optimal(state, end_state):
    return max(euclid_tiles(state, end_state),
               rowcol_tiles(state, end_state)) + getLinear_conflict(state, end_state)

register_heuristic('Proposed (a) max(E,RC)+LC', proposed_optimal)

# (b) FAST: w * (max(E, RC) + LC), w > 1
# not admissible, but solution cost <= w * optimal cost (weighted A* bound)
W_FAST = 1.5

def proposed_fast(state, end_state):
    return W_FAST * proposed_optimal(state, end_state)

register_heuristic(f'Proposed (b) {W_FAST}*(a)', proposed_fast)


# ===== CELL 2: experiment over random states (spiral goal) =====
def experiment(names, n_states=200, size=3, seed=0):
    random.seed(seed)
    end_state = generate_spiral_state(size)
    stats = {n: {'expanded': [], 'suboptimal': 0, 'extra_cost': []} for n in names}
    for _ in range(n_states):
        s = generate_random_state(size, end_state)
        with contextlib.redirect_stdout(io.StringIO()):   # silence "executed" prints
            ref = run_algorithm('A (MD + LC)', s, end_state, size)['A (MD + LC)']
            for n in names:
                r = run_algorithm(n, s, end_state, size)[n]
                stats[n]['expanded'].append(r['nodes_expanded'])
                stats[n]['extra_cost'].append(r['cost'] - ref['cost'])
                stats[n]['suboptimal'] += r['cost'] > ref['cost']
    print(f"{'Algorithm':32s}{'mean expanded':>15s}{'suboptimal':>12s}{'max extra cost':>16s}")
    for n in names:
        st = stats[n]
        print(f"{n:32s}{statistics.mean(st['expanded']):15.0f}"
              f"{st['suboptimal']:9d}/{n_states}{max(st['extra_cost']):16d}")

experiment(['A (MD + LC)', 'Proposed (a) max(E,RC)+LC', f'Proposed (b) {W_FAST}*(a)'])


# ===== CELL 3 (optional): compare several weights for (b) =====
for w in [1.2, 1.5, 2.0]:
    register_heuristic(f'(a) x {w}',
                       (lambda w: lambda s, e: w * proposed_optimal(s, e))(w),
                       include_in_all=False, replace=True)
experiment(['Proposed (a) max(E,RC)+LC', '(a) x 1.2', '(a) x 1.5', '(a) x 2.0'])


# ===== CELL 4: admissibility / consistency checks for (a) =====
initial_state = [7, 8, 1, 4, 0, 6, 2, 3, 5]
end_state = generate_spiral_state(3)
print("h(goal) =", proposed_optimal(end_state, end_state))
print("h(start) =", proposed_optimal(initial_state, end_state))
res = check_consistency(proposed_optimal, initial_state, end_state, 3, max_states=None)
print(res)
if res['counterexample']:
    c = res['counterexample']
    print("violation h(n) - (1 + h(n')) =", c['h_state'] - (1 + c['h_child']))
