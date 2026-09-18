"""Optional minorminer comparison arm. Never called by the learned constructor."""
import time


def baseline_proposal(task, budget, seed, seconds_left, tries=2, router_seconds=2.0):
    # Imports are confined to the explicitly requested comparison path.
    from seeded_minorminer import attempt
    from _context import host_context
    from isingfold.rl.contracts import DecisionState, Opcode
    from isingfold.rl.env import EmbeddingEnv, Mode, fixed_strength_selector

    started = time.monotonic()
    chains = attempt(task, None, seed, tries, budget=budget,
                     timeout=min(seconds_left, router_seconds))
    if chains is None or time.monotonic() - started >= seconds_left:
        return None
    # Compile/validate the baseline's own answer through the same final gate.
    # This wrapper only validates/compiles the returned baseline embedding. It
    # offers no improvement or restart actions and needs no restart-cache bank.
    env = EmbeddingEnv(task, host_context(budget, quotas={}), mode=Mode.IMPROVEMENT,
                       initializer=lambda *_: chains, selector=fixed_strength_selector(), reward_reads=8)
    dec = env.reset(seed)
    if not isinstance(dec, DecisionState):
        return None
    choices = [i for i, (c, ok) in enumerate(zip(dec.candidates, dec.legal_mask))
               if ok and c.opcode is Opcode.COMMIT]
    if not choices:
        return None
    terminal = env.step(dec, choices[0], evaluate_training_reward=False).next_decision_or_terminal
    return terminal if time.monotonic() - started <= seconds_left else None
