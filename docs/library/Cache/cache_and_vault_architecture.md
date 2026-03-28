# Cache and Vault Architecture Overview

> [!summary]
> The deeper combined page was split to reduce bloat. Use the dedicated subsystem docs instead.

Read next:

- [[Cache/architecture]] — runtime cache ownership, lifecycle, invalidation, materialization, and live refresh
- [[Vault/architecture]] — working-vault structure, save/load rules, snapshots, and deployment handoff

Use this rule of thumb:

- if the thing changes because candles changed, start with the **Cache** docs
- if the thing is persisted selected/fitted model state, start with the **Vault** docs

Related:

- [[Cache/user_guide]]
- [[Vault/user_guide]]
