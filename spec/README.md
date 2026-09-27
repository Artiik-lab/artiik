# Conformance fixtures

Language-neutral transcripts that every artiik implementation must convert without loss. The Python package uses them now, and the TypeScript package will reuse them.

Each file in `fixtures/<format>/` is one JSON document:

| Field | Meaning |
|---|---|
| `format` | `anthropic-messages`, `openai-responses` or `openai-chat`. |
| `description` | What the transcript exercises. |
| `covers` | Tags for the cases it covers, such as `parallel-tool-calls` or `thinking`. |
| `messages` | The request's `messages` (Anthropic Messages and Chat Completions). |
| `input` | The request's `input` items (Responses). |
| `system` | Optional: the Anthropic `system` parameter. |

## The rules

1. Parsing a fixture and dumping it back gives JSON equal to the original.
2. Blocks and items kept whole (compaction blocks, encrypted reasoning, block types the implementation doesn't model) serialize byte for byte as they came in.
