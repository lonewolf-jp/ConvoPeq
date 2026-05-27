# ISR R11-R25 Gate Coverage Strategy

作成日: 2026-05-27
対象: `doc/work/ISR_Completeness_Risk_Backlog.md` の R11〜R25（Closed項目）

---

## 1. 目的

R11〜R25 は backlog 上 `Closed` であるため、
「Closed最小検証項目」が CI 実行導線と結びついていることを継続検証する。

本方針は、仕様の再議論ではなく **配線・検証導線の退行検知** を目的とする。

---

## 2. R11-R25 と検証ゲートの対応

| リスクID | 最小検証の主ゲート | 補助ゲート |
| --- | --- | --- |
| R11 Closure Descriptor | `.github/scripts/isr-verify-v3.ps1` | `isr-verify-v10-ownership-cycle.ps1` |
| R12 Payload Tier | `.github/scripts/isr-verify-v4.ps1` | `isr-verify-v10.ps1` |
| R13 Immutable Facade | `.github/scripts/isr-verify-p3-governance.ps1` | `isr-verify-v1-immutability.ps1` |
| R14 Deferred Retire Intent | `.github/scripts/isr-verify-p3-governance.ps1` | `check-list-compliance.ps1` |
| R15 Shutdown FSM | `.github/scripts/isr-verify-v6.ps1` | `isr-verify-v6-domain-f-ordering.ps1` |
| R16 HB Reorder Simulation | `.github/scripts/isr-verify-v5.ps1` | `isr-verify-v2-seal.ps1` |
| R17 Epoch Abstraction | `.github/scripts/isr-verify-v7.ps1` | `isr-verify-v8-shared-split-readiness.ps1` |
| R18 CI Verification Pipeline | `.github/workflows/isr-verification.yml` | `isr-verify-gate-wiring.ps1` |
| R19 capability-first | `.github/scripts/isr-verify-p3-governance.ps1` | `isr-verify-v5-retire-authority-lane.ps1` |
| R20 Host Chaos Normalization | `.github/scripts/isr-verify-p3-governance.ps1` | `isr-verify-v6-domain-f-ordering.ps1` |
| R21 DSP ownership 統合 | `.github/scripts/isr-verify-p3-governance.ps1` | `isr-verify-v4-dsp-handle-policy.ps1` |
| R22 callback-local snapshot | `.github/scripts/isr-verify-v8.ps1` | `isr-verify-v6-domain-f-ordering.ps1` |
| R23 2-world固定 | `.github/scripts/isr-verify-p3-governance.ps1` | `isr-verify-v5-retire-authority-lane.ps1` |
| R24 bounded teardown | `.github/scripts/isr-verify-v6.ps1` | `isr-verify-v7.ps1` |
| R25 DebugRuntime CI限定 | `.github/scripts/isr-verify-runtime-reduction-gate.ps1` | `isr-verify-proof-scope.ps1` |

---

## 3. 運用ルール

- R11〜R25 の各項目は、上表の主ゲートが workflow 内で実行されることを必須とする。
- 新規 rename / alias / 分割により workflow 参照が変更された場合、
  `isr-verify-r11-r25-closed-coverage.ps1` を同時更新する。
- 本方針は「Closed解除」ではなく「Closedの継続妥当性維持」を目的とする。

---

## 4. 自動検証

- `.github/scripts/isr-verify-r11-r25-closed-coverage.ps1`
  - workflow への必須ゲート配線を検証
  - 方針文書の対応表が存在することを検証
