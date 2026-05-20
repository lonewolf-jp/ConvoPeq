# ConvoPeq ISR Proof Artifact Schema Registry

## 目的

本書は R18（CI Verification Pipeline）の正本補助として、
proof artifact の **命名規約** と **JSON schema 雛形** を固定する。

目的は「runtime emits proof / CI verifies proof」を実装班が即時着手できる粒度に落とすこと。

---

## 適用範囲

- `doc/work/ISR_Verification_Pipeline.md` の V2〜V10
- runtime/validator が生成する JSON artifact
- CI evaluator が読み取る schema contract

---

## 命名規約（canonical）

### ルール

- 形式: `snake_case.json`
- 接尾辞:
  - 構造証跡: `_graph.json`
  - 時系列証跡: `_trace.json` / `_timeline.json`
  - 検証レポート: `_report.json`
- 1 validator = 1 primary artifact（必要に応じて secondary artifact を許可）

### Canonical Artifact 一覧

| Validator / Runtime | Canonical artifact | 用途 |
| --- | --- | --- |
| ClosureValidator | `closure_graph.json` | publish graph の closure 証跡 |
| PayloadTierValidator | `payload_tier_report.json` | tier/capability 検証結果 |
| HBVerifier | `hb_constraint_report.json` | HB 制約検証の要約 |
| HBVerifier | `hb_graph_trace.json` | HB runtime trace |
| ShutdownVerifier | `shutdown_trace.json` | shutdown phase/barrier trace |
| RetireAudit | `retire_timeline.json` | retire lane 時系列証跡 |
| RetireAudit | `retire_latency_report.json` | retire latency 集計 |
| SealIntegrityCheck | `mutation_fault_trace.json` | post-publish mutation 違反 trace |
| UAFDetector | `asan_report.txt` | sanitizer 原本ログ（非JSON） |

### 互換 alias（移行期間）

| Legacy | Canonical |
| --- | --- |
| `runtime_publishworld_graph.json` | `closure_graph.json` |
| `retire_lane_timeline.json` | `retire_timeline.json` |

移行完了後は alias 出力を廃止する。

---

## 共通 JSON Envelope

すべての JSON artifact は以下共通フィールドを必須とする。

```json
{
  "schemaVersion": "1.0.0",
  "artifactType": "closure_graph",
  "producer": "ClosureValidator",
  "producedAt": "2026-05-20T12:34:56Z",
  "build": {
    "gitSha": "<commit>",
    "branch": "main",
    "configuration": "Debug"
  },
  "data": {}
}
```

---

## JSON Schema 雛形（Draft 2020-12）

### 1) closure_graph.json

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "isr://schemas/closure_graph/1.0.0",
  "type": "object",
  "required": ["schemaVersion", "artifactType", "producer", "producedAt", "build", "data"],
  "properties": {
    "schemaVersion": { "const": "1.0.0" },
    "artifactType": { "const": "closure_graph" },
    "producer": { "type": "string", "minLength": 1 },
    "producedAt": { "type": "string", "format": "date-time" },
    "build": {
      "type": "object",
      "required": ["gitSha", "branch", "configuration"],
      "properties": {
        "gitSha": { "type": "string", "minLength": 7 },
        "branch": { "type": "string" },
        "configuration": { "type": "string" }
      },
      "additionalProperties": false
    },
    "data": {
      "type": "object",
      "required": ["root", "nodes", "edges", "validationResult"],
      "properties": {
        "root": { "type": "object" },
        "nodes": { "type": "array", "items": { "type": "object" } },
        "edges": { "type": "array", "items": { "type": "object" } },
        "validationResult": { "type": "string" }
      },
      "additionalProperties": true
    }
  },
  "additionalProperties": false
}
```

### 2) payload_tier_report.json

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "isr://schemas/payload_tier_report/1.0.0",
  "type": "object",
  "required": ["schemaVersion", "artifactType", "data"],
  "properties": {
    "schemaVersion": { "const": "1.0.0" },
    "artifactType": { "const": "payload_tier_report" },
    "data": {
      "type": "object",
      "required": ["summary", "violations"],
      "properties": {
        "summary": { "type": "object" },
        "violations": { "type": "array", "items": { "type": "object" } }
      }
    }
  }
}
```

### 3) hb_constraint_report.json / hb_graph_trace.json

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "isr://schemas/hb_constraint_report/1.0.0",
  "type": "object",
  "required": ["schemaVersion", "artifactType", "data"],
  "properties": {
    "schemaVersion": { "const": "1.0.0" },
    "artifactType": { "enum": ["hb_constraint_report", "hb_graph_trace"] },
    "data": {
      "type": "object",
      "required": ["constraints", "result"],
      "properties": {
        "constraints": { "type": "array", "items": { "type": "object" } },
        "result": { "type": "string" }
      }
    }
  }
}
```

### 4) shutdown_trace.json

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "isr://schemas/shutdown_trace/1.0.0",
  "type": "object",
  "required": ["schemaVersion", "artifactType", "data"],
  "properties": {
    "schemaVersion": { "const": "1.0.0" },
    "artifactType": { "const": "shutdown_trace" },
    "data": {
      "type": "object",
      "required": ["phases", "verificationResult"],
      "properties": {
        "phases": { "type": "array", "items": { "type": "object" } },
        "verificationResult": { "type": "string" }
      }
    }
  }
}
```

### 5) retire_timeline.json / retire_latency_report.json

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "isr://schemas/retire_timeline/1.0.0",
  "type": "object",
  "required": ["schemaVersion", "artifactType", "data"],
  "properties": {
    "schemaVersion": { "const": "1.0.0" },
    "artifactType": { "enum": ["retire_timeline", "retire_latency_report"] },
    "data": {
      "type": "object",
      "required": ["summary"],
      "properties": {
        "summary": { "type": "object" },
        "events": { "type": "array", "items": { "type": "object" } },
        "metrics": { "type": "object" }
      }
    }
  }
}
```

### 6) mutation_fault_trace.json

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "isr://schemas/mutation_fault_trace/1.0.0",
  "type": "object",
  "required": ["schemaVersion", "artifactType", "data"],
  "properties": {
    "schemaVersion": { "const": "1.0.0" },
    "artifactType": { "const": "mutation_fault_trace" },
    "data": {
      "type": "object",
      "required": ["faultCount", "faults"],
      "properties": {
        "faultCount": { "type": "integer", "minimum": 0 },
        "faults": { "type": "array", "items": { "type": "object" } }
      }
    }
  }
}
```

---

## CI 評価ルール（schema contract）

以下はいずれも CI fail:

- artifact missing
- JSON parse error
- schema mismatch
- `schemaVersion` 不一致
- `artifactType` とファイル名の不一致

---

## ステータス

- Spec-Fixed: 2026-05-20
- Closed: 未完（schema validator 実装と CI 接続が未完）
