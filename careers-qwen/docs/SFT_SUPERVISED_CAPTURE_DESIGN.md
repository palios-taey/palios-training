# Supervised Taey capture seat

Design only: this producer contract creates no training rows and does not
reconstruct history. The private raw store keeps exact bytes; the public export
keeps redacted metadata and hashes.

## Common event record

Each append-only event carries `schema_version`, `trace_id`, `sequence`,
`recorded_at`, `request_sha256`, actor, source reference, and exact payload
hashes. The request stores the exact system/user messages. A model proposal
stores the exact tool name, arguments, `call_id`, and sequence. A result stores
exact stdout, stderr, structured result bytes, exit status, and hashes.

## Read-only path

`request -> model proposal -> read-only tool executes -> exact result -> next
model decision -> independent validation`. Read-only Git inspection may use the
existing authority without extra approval. Missing or reordered events refuse
admission rather than being filled from logs or memory.

## State-changing path

`request -> model proposal -> supervisor approval -> execute once -> exact result
-> next model decision -> independent validation -> CONTROL/merge receipt`.
The approval record is bound to the exact `call_id` and SHA-256 of the exact
arguments. The executor physically refuses the operation unless a matching,
unused approval exists; approval cannot be reused or broadened. This applies to
Git commit/push/PR/review/merge and dispatch, notify, task-closure, or other
state-changing operations.

## Public-safe export and admission

Export trace IDs, sequence, operation class, commit/PR IDs, result hashes,
validation status, and approval/closure links. Remove credentials, hostnames,
filesystem paths, private topology, and private request content. Admit a trace
only when request, proposal, exact result, next decision, validation, and (for
state changes) approval plus CONTROL closure are complete. This design does not
authorize sanction, mixture, dose, or a training launch.
