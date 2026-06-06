# RuntimeWorld Decision Input Inventory (0-10)

作成日: 2026-06-06
ベース: RuntimeState (AudioEngine.h:121-)

---

## RuntimeState 全フィールドの Decision 入力利用状況

| フィールド | AuthorityClass | 現在のDecision入力利用 | 将来のDecision化可能性 |
|---|---|---|---|
| `worldId` | Diagnostic | なし (診断用) | なし |
| `generation` | Authoritative | なし (世代管理) | 低 |
| `generationSemantic` | Derived | なし | 低 |
| `topology` | Authoritative | なし | 中 (ルーティングDecision) |
| `routing` | Authoritative | なし | 中 (ルーティングDecision) |
| `execution` | Authoritative | なし (遷移ポリシー格納) | 中 (実行時Decision) |
| `publication` | Authoritative | なし | 中 |
| `overlap` | Authoritative | CrossfadeDecisionの出力格納先 | 低 (出力専用) |
| `metadata` | Authoritative | なし | 低 |
| `retire` | Authoritative | なし | 低 |
| `timing` | Authoritative | なし (fadeTimeSec格納) | 中 |
| `latency` | Authoritative | なし | 低 |
| `graph` | Derived | なし | 低 |
| `engine` | Derived | なし | 低 |
| `resource` | Derived | なし | 中 (リソース制約Decision) |
| `affinity` | Diagnostic | なし | なし |
| `automation` | Derived | なし | 中 (自動化Decision) |
| `coefficient` | Derived | なし | 低 |
| **`dspProjection`** | Derived | **CrossfadeAuthority のみ** | **高 (全Decisionの基盤)** |
| `projectionFreshness` | Diagnostic | なし | なし |
| `semanticHash` | Diagnostic | なし | なし |

## 結論

- CrossfadeAuthority 以外の Decision が DSPCore を直読するリスクは現在のところ確認されていない
- 将来新たな Decision が追加される場合、`dspProjection` が唯一の決定入力として適切
- 新規 Decision 作成時は `RuntimeWorld.dspProjection` 経由で値を取得する規約を徹底する
