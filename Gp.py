"""
Genetic Programming for Paper‑Reviewer Assignment — Pure‑Python
==============================================================
**New feature:** *Intentionally bad Generation 0* to observe GP’s ability to
improve from a poor starting point.

Pass the command‑line flag `--bad_init` (or `-b`) and the script will seed the
entire initial population with **constant‑score trees** that make the greedy
builder always pick the *first feasible reviewer* ⇒ typically yields a wildly
unbalanced load (high `max_load`).

```bash
python gp_no_lib.py hustack2.txt --bad_init         # worst‑case start
python gp_no_lib.py hustack2.txt                    # normal random start
```

Implementation notes
--------------------
* **Bad tree generator** returns `('term', 'const', C)` where `C` ≫ 0
  (here C ∈ [50, 100]).  Because every reviewer gets identical score, the tie
  is broken by the first candidate in the list, stressing the algorithm.
* You can easily craft other “bad factories” — e.g. constant negative scores
  (always last candidate), or pathological arithmetic trees — just tweak
  `gen_bad_tree()`.

Everything else (fitness, operators) stays unchanged.
"""

from __future__ import annotations

import copy
import math
import random
import sys
from typing import Any, List, Tuple

################################################################################
#                               ── CLI args ──                                 #
################################################################################

args = sys.argv[1:]
BAD_INIT = True
if "--bad_init" in args or "-b" in args:
    BAD_INIT = True
    args = [a for a in args if a not in ("--bad_init", "-b")]

inst_path = args[0] if args else "hustack5.txt"

################################################################################
#                               ── Instance ──                                 #
################################################################################

def read_instance(path: str) -> Tuple[int, int, int, List[List[int]]]:
    with open(path, "r", encoding="utf-8") as f:
        tok = [int(x) for line in f for x in line.strip().split()]
    N, M, b = tok[:3]
    idx = 3
    L: List[List[int]] = []
    for _ in range(N):
        k = tok[idx]; idx += 1
        revs = tok[idx:idx+k]; idx += k
        L.append([r-1 for r in revs])
    return N, M, b, L

N, M, B, L = read_instance(inst_path)
print(f"Loaded {inst_path}: N={N}, M={M}, b={B}, BAD_INIT={BAD_INIT}")

GLOBAL_QUOTA = math.ceil(N*B/M)
REVIEWER_DEG = [0]*M
for i in range(N):
    for r in L[i]:
        REVIEWER_DEG[r]+=1

################################################################################
#                               ── GP core ──                                  #
################################################################################

OPS = {
    'add': lambda a,b: a+b,
    'sub': lambda a,b: a-b,
    'mul': lambda a,b: a*b,
    'div': lambda a,b: a/b if abs(b)>1e-9 else a,
    'min': lambda a,b: a if a<b else b,
    'max': lambda a,b: a if a>b else b,
}
OP_NAMES=list(OPS.keys())
TERM_NAMES=['load','slack','deg','candCnt','rand','const']

def rev_idx(x):
    return int(abs(x))%M

def pap_idx(x):
    return int(abs(x))%N

def eval_tree(node:Any,r:int,i:int,loads:List[int]):
    t=node[0]
    if t=='op':
        _,name,lft,rgt=node
        return OPS[name](eval_tree(lft,r,i,loads),eval_tree(rgt,r,i,loads))
    _,name,val=node
    if name=='load': return loads[rev_idx(r)]
    if name=='slack': return GLOBAL_QUOTA-loads[rev_idx(r)]
    if name=='deg': return REVIEWER_DEG[rev_idx(r)]
    if name=='candCnt': return len(L[pap_idx(i)])
    if name=='rand': return random.random()
    if name=='const': return val
    raise ValueError

MAX_DEPTH=4

def rand_const():
    return random.uniform(-2,2)

def gen_random_tree(depth=0):
    if depth>=MAX_DEPTH or (depth>0 and random.random()<0.3):
        term=random.choice(TERM_NAMES)
        val=rand_const() if term=='const' else None
        return ('term',term,val)
    op=random.choice(OP_NAMES)
    return ('op',op,gen_random_tree(depth+1),gen_random_tree(depth+1))

# --------------- intentionally bad tree ----------------

def gen_bad_tree():
    # constant big positive so tie‑break selects first feasible reviewer
    return ('term','const',random.uniform(50,100))

# operators

def deepcopy(n):
    return copy.deepcopy(n)

def mutate(tree):
    if random.random()<0.1 or tree[0]=='term':
        return gen_random_tree()
    _,name,l,r=tree
    if random.random()<0.5:
        return ('op',name,mutate(l),r)
    return ('op',name,l,mutate(r))

def crossover(a,b):
    if a[0]=='term' or b[0]=='term':
        return deepcopy(b)
    if random.random()<0.5:
        return ('op',a[1],crossover(a[2],b[2]),a[3])
    return ('op',a[1],a[2],crossover(a[3],b[3]))

# assignment and fitness

def build_assignment(f):
    loads=[0]*M
    viol=0
    for i in range(N):
        assigned=[]
        for _ in range(B):
            cands=[r for r in L[i] if r not in assigned]
            if not cands:
                viol+=1; break
            scores=[(eval_tree(f,r,i,loads),r) for r in cands]
            _,best=min(scores,key=lambda t:t[0])
            assigned.append(best)
            loads[best]+=1
    return max(loads),viol

def better(f1,f2):
    return f1<f2

# GP loop params
POP=200
GEN=40
TOUR=5
CXPB=0.9
MUTPB=0.1

# init population
pop=[gen_bad_tree() if BAD_INIT else gen_random_tree() for _ in range(POP)]
fit=[build_assignment(ind) for ind in pop]
bi=min(range(POP),key=lambda j:fit[j])
best=deepcopy(pop[bi]); best_fit=fit[bi]
print(f"Gen0 best {best_fit}")

for g in range(1,GEN+1):
    new=[]
    while len(new)<POP:
        def sel():
            ks=random.sample(range(POP),TOUR)
            ks.sort(key=lambda j:fit[j])
            return deepcopy(pop[ks[0]])
        p1,p2=sel(),sel()
        child=crossover(p1,p2) if random.random()<CXPB else deepcopy(p1)
        if random.random()<MUTPB: child=mutate(child)
        new.append(child)
    pop=new
    fit=[build_assignment(ind) for ind in pop]
    idx=min(range(POP),key=lambda j:fit[j])
    if better(fit[idx],best_fit):
        best_fit=fit[idx]; best=deepcopy(pop[idx])
    if g%1==0 or g==GEN:
        print(f"Gen{g} best {best_fit}")

print("\nBest tree:",best)
print("Fitness (max_load, violations):",best_fit)
