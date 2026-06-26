# Loop — recurring agent prompt

Parse `/loop [interval] <prompt>` per the loop skill pattern.

## Examples

- `/loop 5m run eval_loop and report`
- `/loop 30m check gh pr checks`
- `/loop monitor overnight test progress`

## Rules

1. Run the prompt **once immediately**
2. Arm background watcher or sleep loop with unique sentinel
3. On each tick: execute prompt, brief status update
4. Stop when user asks or task completes

For UploadM8 eval monitoring, prefer:
`python scripts/agent/eval_loop.py --mode unit --json`

See **uploadm8-routines** skill for Automation templates.
