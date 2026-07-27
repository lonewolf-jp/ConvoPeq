了解です。この形で REPAIR_PLAN に反映します。

---

## REPAIR_PLAN 反映内容（修正案）

### B-1（修正）

| 項目 | 内容 |
|------|------|
| **名称** | **Publication Metadata Propagation to Retire Path** |
| **本質** | Publication Metadata を Retire Path まで伝搬する設計が未確定。Authority不足ではない |
| **状況** | 別 Issue として追加設計が必要 |
| **表現** | 「Authority がない」「Owner がいない」ではなく「Retire Path まで伝わらない」 |

### B-2（修正なし）

| 項目 | 内容 |
|------|------|
| **本質** | Queue Protocol の安全性が未確定 |
| **記載** | reservation / payload visibility / commit の責務分離を含めた追加解析課題 |
| **注意** | three-phase protocol を採用案として書かない |

### B-3（修正なし）

| 項目 | 内容 |
|------|------|
| **確認済み** | `updateFade()` が Audio Thread から未呼び出し |
| **断定しない** | 「Snapshot Fade が未動作」は DSP 側全経路解析後に譲る |

---

確認済みの事実と設計課題・仮説が明確に区別された文面になります。