# Running DataWorks

The supported Python version is **3.12**. CI is configured for Windows and Linux;
the recorded acceptance runs use Windows, and Linux execution remains unverified.
The preserved dependency pins installed successfully into a fresh Windows
environment on 2026-10-30; all 559 regression tests and the fresh-process
fulfillment proof passed there. The earlier installed environment also passed.
See [REPRODUCING.md](REPRODUCING.md) for exact observed versions, reproduction
commands, and remaining release gates. Once the selected environment is installed,
run:

```shell
python -m streamlit run app.py
```

The committed configuration binds to `127.0.0.1`, keeps CORS/XSRF protection enabled, and limits uploads to 50 MiB. Authentication is disabled for local use. Work and models stay in the process and browser session until exported; restarting the service does not retain sessions. Save needed projects before restarting or updating.

## Tests

```shell
python -m pip check
python -m compileall -q app.py backend frontend scripts tests
python scripts/run_tests.py
```

The runner selects a disposable repository-local temporary directory before
importing Streamlit tests. No test data is sent to an external service. The suite
covers actual legacy/current Excel parsing, project round trips, cleaning/schema
undo, fitted preprocessing, ML evaluation, and Streamlit page checks. The CI job
is configured to run that suite; its presence does not establish a successful
Linux run. Installation of the preserved pins was separately verified locally.

## Private shared deployment

Run behind a TLS reverse proxy that supports WebSockets and limits request sizes and connection counts. Keep the Streamlit process on loopback; do not expose its port directly. If a container must listen on `0.0.0.0`, restrict its network/ingress to the proxy. Use one process per container and set explicit host/container memory and CPU limits. A process has independent sessions but a shared resource ceiling; run more isolated containers if concurrency requires them.

Enable the built-in OIDC gate before accepting shared traffic:

```text
DATAWORKS_REQUIRE_AUTH=true
DATAWORKS_ALLOWED_EMAILS=analyst@example.com,reviewer@example.com
```

The optional allowlist matches full addresses case-insensitively and requires the provider's `email_verified` claim to be the boolean `true`. If the provider does not return that claim, access with an allowlist fails closed. Configure the provider to issue verified email claims, or enforce account/group restrictions at the provider and omit the email allowlist. Without the allowlist, every account authenticated by the configured provider may use the app.

Create `.streamlit/secrets.toml` on the host using your provider's actual configuration. Never commit this file. The app currently supports Streamlit's default OIDC provider:

```toml
[auth]
redirect_uri = "https://dataworks.example.com/oauth2callback"
cookie_secret = "replace-with-a-strong-random-secret-at-least-32-characters-long"
client_id = "provider-client-id"
client_secret = "provider-client-secret"
server_metadata_url = "https://identity.example.com/.well-known/openid-configuration"
```

Register the exact HTTPS callback URL with the provider. Do not expose identity/access tokens to the app. Generate the cookie secret using a cryptographic generator, for example `python -c "import secrets; print(secrets.token_urlsafe(48))"`. Supply secrets through the host's secret store and rotate them as operationally required. Authentication misconfiguration blocks access before dataset pages load.

The gate uses Streamlit's native `st.login`/`st.user` integration and the pinned Authlib dependency. Local mock tests verify allow/deny and failure behavior. **A real provider callback and the final proxy/TLS configuration must be verified in the target environment before shared use.** See the [Streamlit authentication reference](https://docs.streamlit.io/develop/api-reference/user/st.login) for provider configuration.

## Diagnostics and operation

DataWorks emits JSON events to standard error with a UTC timestamp, severity, fixed event name, numeric counters, and exception class when relevant. App diagnostics omit cell values, column names, upload filenames, identities, exception messages, and tracebacks. Route process logs to your host's log collector and configure retention there; no unbounded application log file is written. Streamlit and third-party libraries have their own logs, so apply the host's normal access controls and retention policy to the entire output.

Use `/_stcore/health` on the private Streamlit port for a process health probe. Monitor process memory, CPU, rejected resource-limit operations, and model failures. Application limits bound common workloads, but they are not a substitute for host/container limits. Avoid running the server with elevated privileges, and keep exported datasets, project files, and trusted model artifacts under the same access controls as the source data.

The included CSV is only a deliberately unclean test fixture. Its upstream provenance/license is not established by this repository; no distribution rights or production data assumptions are inferred from its presence. Project licensing must be chosen by the owner before redistribution.
