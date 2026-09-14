# Local prospective identity audit only

The preflight files below were generated on macOS with a developer environment.
They prove that the current algorithm resolves the complete 3,082-lineage plan and
that its prospective split and problem identities are globally unique in that
environment. They are not publication or launch artifacts.

Do not pass either file to the Apollo or Goose launcher. The authoritative
`prospective_preflight_v2.json` must be generated once in the frozen Linux x86_64
runtime identified by `../SOURCE_RELEASE.json`, then fully replayed on both Apollo
and Goose. The launchers fail closed if its source or provenance differs.

The local environment was CPython 3.12.0 with dimod 0.12.22,
dwave-networkx 0.8.18, dwave-samplers 1.7.0, minorminer 0.2.22,
networkx 3.6.1, numpy 2.4.6, and scipy 1.17.1. Its generation-provenance digest was
`2abfeae468fc7df54a23b3b1d9f7d74662530b126345bf0221de74784d1d6d05`.

The `source-2259928e` directory records the audit before the Goose CPU request was
reduced from 8 CPUs to 1 CPU. The logical identity map is unchanged. The
`source-5f8f636d` directory binds the final source inventory at this checkpoint.
