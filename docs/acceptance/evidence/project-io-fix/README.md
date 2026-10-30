# NumPy project decoder validation fix

Confirmed malformed payloads previously allowed integer truncation (1.5 to 1),
float32 rounding (16777217 to 16777216), fixed-width text truncation, and a
`numpy_scalar` tag returning a non-scalar array.

The decoder now validates numeric types/ranges/precision, string types/width,
and typed time values before assigning them. Scalar tags reject non-scalar input
and object-scalar declarations. Supported object arrays remain unchanged.

Three new regression tests cover invalid payloads and valid round trips,
including NaN/infinity, complex values, strings, time values, object arrays,
and zero-dimensional arrays. The 21 focused codec tests pass in both environments.

`original/results.json` and `declared/results.json` record complete verification:
562 tests pass in each environment, actual fresh-process fulfillment/project
reproduction passes, and source hashes remain stable during checks. The declared
environment matches the preserved pins; the original environment is explicitly
marked with the historical-version override. Source syntax and diff checks pass.

The earlier Phase 6 verification ZIP is an immutable historical artifact and
does not include this later fix. It must be rebuilt and verified before being
used as the package for the updated source. No commit or release was created.

Suggested Git message: `fix: reject lossy NumPy project payloads`.
