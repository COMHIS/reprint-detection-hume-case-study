"""The five tests an evidence layer has to pass before it carries a claim.

Each checks a specific joint in the construction. None of them requires the
models in the rest of this package: a project with no models at all can run all
five, provided its evidence is organised so that they can be run.

1. ``page_image_check``   -- reconstruct events on the destination side and read
                             a sample against page images (procedure, not code).
2. ``null_model``         -- test a distributional claim against a null that
                             preserves the structure of the derivation.
3. ``compiled_units``     -- compare catalogue authority with a signal computed
                             from the objects, in both directions.
4. ``source_ablation``    -- repeat source-side ablations at more than one
                             bibliographical level, conditioned on content.
5. ``coverage_audit``     -- re-test outside the layer that produced a
                             regularity, and bound what the record misses.
"""
