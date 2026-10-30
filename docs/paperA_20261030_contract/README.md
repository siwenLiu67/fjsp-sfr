# Paper A 2026-10-30 contract registry

This directory registers the manuscript sources supplied in
`PaperA_Codex_CC_DVL_Stages_20261030.zip`. The archive is the authoritative
source; repository files under `paper_sections/` and
`tests/PAPER_CONTRACTS_SEC3_5.md` are historical snapshots when they conflict.
The manuscript text is not duplicated here so later author revisions cannot
be mistaken for a silently edited copy. A source is accepted only when its
SHA-256 matches this registry.

| Source | SHA-256 |
| --- | --- |
| `PaperA_Codex_CC_DVL_Stages_20261030.zip` | `444d6c6c37881c20c08b4d87b9f82e2e62d2f92a683e25f7b7e20b032a36f39c` |
| `paper_contract/sec3.tex` | `dc56c9d4ecf47635280da20db2e7f419590b838ccec4beaab9fe6cfbb38cf3e2` |
| `paper_contract/sec4_v2_2r.tex` | `b40f07187440678c8a5c61b3a47899d598045a216ca36b93898f0be103b902f7` |
| `paper_contract/sec5_CC_DVL_fixed_20261030.tex` | `0dd62d626c22a12474b1a8a045f7e274bcd5619f3b5b919b406cbe32149272bb` |
| `paper_contract/sec6_CC_DVL_fixed_20261030.tex` | `7b08914b3f775bcb2748263c3658bf1989846688a1ac7ce719a986be6a7069ed` |

Historical repository hashes at the Phase-0 base commit
`fa9e045616624813afe6a13ae69df2fbc05d10fa` are:

| Historical source | SHA-256 | Relationship |
| --- | --- | --- |
| `paper_sections/sec4.tex` | `5a4bf0bfa9175e9d97be0b5e96e09f77a6d9869c55b845fcd194046159db77f9` | Older standalone section; not the 2026-10-30 Sec.4 contract. |
| `paper_sections/sec5.tex` | `a7ab26ef615338a6d88ebeeca5ab1bbeb06f3fd3c69cc327710e94ac5f6f1ae4` | Older HGT manuscript; conflicts with the CRG/CC-DVL Sec.5 contract. |
| `tests/PAPER_CONTRACTS_SEC3_5.md` | `d7feb2135cd12196e9b41f3be630934fc959b3e4894cbdcd0f6fdd918c9df140` | Historical contract through the prior paper14/paper_graph stage. |

The executable integration-path declaration is
`experiments/paperA_contract.py`; the detailed equation-to-code inventory is
`docs/paperA_20261030_contract_matrix.md`.
