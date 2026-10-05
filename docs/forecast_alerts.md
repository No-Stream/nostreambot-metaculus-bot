# Forecasting failure explanations

Forecasting workflows retain the bot's existing exit codes, Bash `pipefail`, source degradation counters, and Playwright installation warning. The bot writes `run_logs/status.json` when `FORECAST_RUN_STATUS_PATH` is set, recording measured source losses, model outcomes, and publication attempts independently of its final exit decision. The always-run summary step renders that evidence into the GitHub run page and emits an error annotation for degraded or failed runs. Missing evidence appears as unknown, including an early setup failure, a killed process, or malformed JSON. The renderer uses system Python and its standard library, so dependency installation does not need to succeed first.

Source causes distinguish logical HTTP lookups from per-question source checks. For example, three questions can share one failed PredictIt catalogue request, producing one failed lookup and three affected source checks. These counts describe separate observations and should not be summed. Source checks also explain failures when an HTTP 200 response contains malformed provider data. The overall outcome follows the existing CLI alert policy; observed source losses can accompany a clean exit when that policy tolerates them, and the summary explains this explicitly. Optional Manifold detail losses retain their existing policy: partial detail loss is tolerated when another detail succeeds, while total detail loss remains alertable.

A downstream `report_status` job uses the compact summary as its display name and runs with `if: always()`, including after a forecasting failure. It fails when the forecasting job did not succeed, while leaving that job's original failure intact. If GitHub withholds the output or the runner never writes it, the display name explicitly says that the forecast status is unknown. Job timeouts and whole-workflow cancellation can prevent finalization, artifact upload, or reporting altogether; an absent report does not establish a successful publication. The summary records successful HTTP publication responses, not an independent readback of the public platform.

## Example and native email verification

The synthetic fixture in `tests/fixtures/forecast_alert_degraded.json` represents a run that published successfully while losing a Polymarket lookup to HTTP 403. The notification-only workflow `forecast_notification_test.yaml` renders it and deliberately fails both its synthetic forecasting job and its reporting job. It receives no secrets, invokes no forecasters, performs no research, and publishes no predictions or comments. Its displayed job name and summary provide an example of the proposed notification text, rather than evidence of a real forecast run. Dispatching it is separate from dispatching any live bot workflow.

The renderer produces this job name and run-page summary from the synthetic fixture:

```text
Published OK; Polymarket HTTP 403

Forecast run: Degraded
Cause: Polymarket HTTP 403; 1/13 lookups affected; other outcomes unknown.
Models: 3/3 succeeded, 0 timed out, 0 dropped.
Forecasts: 1/1 succeeded, 0 failed.
Comments: 1/1 succeeded, 0 failed.
```

GitHub documents [job names, outputs, and `always()` after upstream failure](https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-syntax), and its [context table](https://docs.github.com/en/actions/reference/workflows-and-actions/contexts) permits `needs` in job names. Its [workflow notification documentation](https://docs.github.com/en/actions/concepts/workflows-and-actions/notifications-for-workflow-runs) describes notification settings but does not promise an email template or inclusion of dynamically named downstream jobs. Local tests verify the producer, renderer, workflow wiring, and reporting shell behavior; they cannot establish what appears in a recipient's native email. GitHub also requires a manually dispatched workflow to exist on the default branch, so this new fixture workflow cannot be dispatched before merge; see [manual workflow requirements](https://docs.github.com/en/actions/how-tos/manage-workflow-runs/manually-run-a-workflow). Email rendering therefore remains unverified until someone receives and inspects the notification-only workflow's failure email. No separate notification service or new credentials are introduced.
