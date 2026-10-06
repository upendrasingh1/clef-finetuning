# Vendored upstream code

`joint_schema_model.py` is Cloudflare's Clef inference code, copied **unmodified**. Do not edit it; put changes
in `clef_finetuning.models` instead, so the diff between upstream and this project stays obvious.

| Field | Value |
| --- | --- |
| Source | https://huggingface.co/Cloudflare/clef-flash/blob/17f0b0ad64efb65d273590632833508766b2aae6/joint_schema_model.py |
| Revision | `17f0b0ad64efb65d273590632833508766b2aae6` |
| SHA-256 | `0e304cf7c6500e8bb59bef7e2afd2c6373f82596dfb3b57d1aa93c175e2dc3a3` |
| License | Apache-2.0 (Cloudflare, Inc.) |

The file is byte-identical in `Cloudflare/clef` at revision `2f3de3dd85f379784083b0814d997ab627200f0c`.
`tests/test_upstream_pin.py` fails if the file changes.
