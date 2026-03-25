# Documentation Map

This folder holds the cross-project documentation for Project NUEVO.

The goal of this index is to make it clear which document owns which topic, so
the repo does not drift into multiple competing "source of truth" files.

## Current Sources of Truth

| Topic | Primary document | Notes |
|---|---|---|
| Logical UART/TLV protocol design | [`COMMUNICATION_PROTOCOL.md`](COMMUNICATION_PROTOCOL.md) | Human-readable message semantics, framing, bootstrap, and streaming policy |
| Exact TLV payload layouts | [`../tlv_protocol/TLV_Payloads.md`](../tlv_protocol/TLV_Payloads.md) | Byte-level payload definitions and sizes |
| Numeric TLV IDs | [`../tlv_protocol/TLV_TypeDefs.json`](../tlv_protocol/TLV_TypeDefs.json) | Machine-readable type registry |
| Cross-project conventions | [`DESIGN_GUIDELINES.md`](DESIGN_GUIDELINES.md) | Naming, numbering, source-of-truth rules, protocol update workflow |
| Arduino firmware architecture | [`../firmware/README.md`](../firmware/README.md) | Firmware overview and firmware doc map |
| Firmware subsystem details | [`../firmware/docs/README.md`](../firmware/docs/README.md) | Index of focused firmware implementation notes |

## Hardware References

The PDFs in this folder are hardware/vendor references used during firmware and
board work:

- `A4988.pdf`
- `ICM-20948.pdf`
- `PCA9685.pdf`
- `dc_motor_specs.pdf`
- `lm61460-q1.pdf`

Treat them as reference material, not repo-specific source-of-truth documents.
