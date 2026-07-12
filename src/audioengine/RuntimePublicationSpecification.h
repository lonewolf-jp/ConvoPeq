#pragma once

// ★ v8.3: RuntimePublishSpecification は RuntimeBuilder.h で定義。
//   本ファイルは RuntimePublicationSpecification.h → RuntimeBuilder.h のエイリアス。
//   依存方向: Orchestrator → Specification(RuntimeBuilder.h) → Builder
//   DSPCore が AudioEngine のネストクラスであるため、実体定義は
//   AudioEngine.h を include する RuntimeBuilder.h に配置されている。

#include "RuntimeBuilder.h"
