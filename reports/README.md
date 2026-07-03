# Reports

This folder contains **small, publishable** report files that back claims in the
README and docs without requiring you to execute the heavy `problems/` notebooks
just to see what a full run looks like.

Important notes:

- These are **reference numbers** from a particular environment, on a particular
  date. Your results may differ slightly due to random seeds, CPU/GPU kernels,
  or upstream dataset availability.
- This repo intentionally does **not** track raw datasets or model weights. To
  reproduce any report from scratch, follow the corresponding notebook in
  `problems/` and re-generate the metrics locally.

Files:

- `case_studies_summary.csv`: the metrics table shown in the README and
  `docs/07-evaluation.md` for the eight `problems/` case studies.

