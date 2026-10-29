# Rendered browser and keyboard smoke

Performed 2026-10-29 using the computer-use skill and Codex in-app browser against
`http://127.0.0.1:8515`. This is an engineering smoke check, not an observation of
an operations analyst or an accessibility conformance audit.

The [local harness](browser_app.py) renders the actual fulfillment/project pages,
StateManager, and backend. Its labeled sidebar loads the original synthetic
fixtures directly; file-dialog/upload transport and actual dashboard import are
outside this browser check. AppTest separately exercises the actual upload
boundary with fixture bytes and actual archive bytes.

The first sandboxed server attempt could not serve local sockets (a socket probe
returned Windows error 10013; initial browser connection was refused). The same server
command was then authorized outside the sandbox, bound only to loopback. No
network/security settings were changed. The server was restarted after the final
wording changes to inspect the final code. Both servers and the temporary browser
tab were stopped/closed after verification.

Observed final-code sequence:

1. Load Month A using the synthetic transport; run the actual page action. Check
   12 source lines, four accepted, eight quarantined, three accepted orders,
   quantity five, and the current successful-run association.
2. Inspect the native table toolbar. It exposes its own preview CSV download;
   the final captions explicitly distinguish that file from the full-result
   fulfillment download buttons. Exact issue JSON now starts collapsed.
3. Select/load the unseen-status fixture and run. The current source changes to
   prefix `ea6ede398824`, while the earlier successful A source is still
   `87f4eaeb0d07`, run prefix `828a1716223f`. The page shows decision required,
   unmapped `backordered`, and an unselected meaning with Apply disabled.
4. Open the meaning control using the Down key, navigate with an arrow key and
   Enter, and visibly verify the selected value. For this synthetic interaction
   the final choice was `fulfilled`; this is not a recommended business meaning
   for actual backorders. Automated journeys independently exercise `pending`
   and `cancelled` choices and their resulting accepted values.
5. Tab to **Apply reviewed status**, verify keyboard focus on that button, and
   press Enter. Plan prefix `d898a4e099aa` advances from revision 1 to revision 2;
   decision controls disappear. The latest attempt becomes **Not run**, and the
   prior A result remains associated with revision 1. Applying a decision did not
   execute the file or replace the report.

[The captured accessibility state](browser-after-decision.txt) records that final
association. [The decision image](browser-decision.png) shows the explicit choice
and retained earlier result; [the post-decision image](browser-workspace.png)
shows no execution for the revised inputs and the earlier report totals.

The browser was a narrow in-app viewport; responsive sections stacked vertically.
This verifies the exercised control names, arrow/Tab/Enter path, rendering, and
state transition only. It does not verify every keyboard path, a screen reader,
all viewport sizes, real-file dialog behavior, user comprehension, or preparation
time. No actual operator, customer file, login, or external upload was involved.
