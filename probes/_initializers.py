"""The two initializers a Track B episode can be protected by, behind one name.

The environment hands the policy a starting embedding and guarantees it can always return that
embedding unchanged, so the initializer sets the floor the policy is measured from. Until now
that floor has always been `router_initializer`, a self-contained greedy placer written so the
package does not depend on minorminer. `runs/external_baseline.log` showed what that costs: the
router floor sits below stock minorminer, and a policy that improves on a weak floor can still
lose to the tool it was meant to replace.

`minorminer` here is the same call the field would make, with isolated variables placed after the
fact because `find_embedding` only returns variables that carry an edge.
"""
from isingfold.rl.proposal import router_initializer


def minorminer_initializer(tries: int = 10):
    import minorminer

    def _init(logical, host, seed):
        isolated = [v for v in logical.nodes() if logical.degree(v) == 0]
        emb = minorminer.find_embedding(list(logical.edges()), list(host.edges()),
                                        random_seed=seed % (2 ** 31), tries=tries)
        if not emb or set(emb) != set(logical.nodes()) - set(isolated):
            return None
        chains = {v: frozenset(c) for v, c in emb.items()}
        used = {q for c in chains.values() for q in c}
        free = iter(sorted(set(host.nodes()) - used))
        for v in isolated:
            try:
                chains[v] = frozenset([next(free)])
            except StopIteration:
                return None
        return chains
    return _init


def pick_initializer(name: str, tries: int = 10):
    if name == "router":
        return router_initializer()
    if name == "minorminer":
        return minorminer_initializer(tries)
    raise ValueError("initializer must be 'router' or 'minorminer', got %r" % name)
