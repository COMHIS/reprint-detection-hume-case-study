# Reprint Detection: A Hume Case Study

Code for two connected studies of essay-scale republication and reuse in
eighteenth-century books and newspapers. They are consecutive stages of one
problem, so they share this repository and are released under separate tags.

| Directory | Paper | Stage |
| --- | --- | --- |
| [`cikm-pair-level/`](cikm-pair-level) | CIKM '26 short paper | Which source–destination **pairs** stand in a plausible transmission relation, given fragmented reuse hits |
| [`dsh-event-level/`](dsh-event-level) | DSH article | What has to be constructed to turn those pairs into historical **events**, and the tests the result must pass |

A pair is not an event. A pair records that one source essay in one digitised
manifestation stands in a plausible relation to one destination document; an
event records that a particular passage of that destination re-offered the text
to a reader. One destination may hold several events, one event may be
supported by many pairs, and a pair may be a real textual relation that is not
a republication at all. The second directory is about that gap.

Each directory is self-contained: its own `README.md`, `DATASETS.md`,
`requirements.txt` and examples. Neither depends on the other at runtime.

## Corpora

Both studies draw on *Eighteenth Century Collections Online* (ECCO), Parts I and
II, and on Gale's Burney and Nichols newspaper collections. OCR text, page
images and subscription-corpus content are licensed and are not distributed
here. Each directory documents the schemas it expects and ships one small
example file per schema, so the pipelines can be rebuilt inside a licensed
environment.

Every stage in both directories preserves half-open character offsets on both
sides, keeps the analytic units distinct with explicit conversions between
them, and appends corrections rather than overwriting an earlier stage.

## Citation

For the pair-level stage:

```bibtex
@inproceedings{shu2026pairlevel,
  author    = {Shu, Ke and Hinderks, Kira and M\"{a}kel\"{a}, Eetu and Tolonen, Mikko},
  title     = {Pair-Level Essay-Scale Republication and Reuse from Fragmented Historical Text Reuse: A Workflow Study on Eighteenth-Century Books and Newspapers},
  booktitle = {Proceedings of the 35th ACM International Conference on Information and Knowledge Management (CIKM '26)},
  year      = {2026},
  doi       = {10.1145/3799682.3839901}
}
```

The event-level citation is added when the DSH article is published.

## License

MIT; see `LICENSE`.
