"""Self-contained verifier code synchronized beside the WP4a paper entrypoints.

This package deliberately lives under ``scripts/`` so adding or updating a paper checker does
not mutate the registered historical training-source digest.  Its transitive Python bytes are
bound by ``audit_source_sha256``.
"""
