#pragma once

#include <cassert>

#include "AlignedAllocation.h"
// RuntimeState は .cpp で完全型をinclude（グローバル名前空間で定義）

struct RuntimeState; // 前方宣言（グローバル名前空間）

namespace convo {

// ★ Phase4 (ALTERNATIVE DESIGN — Builder/Runtime 二段階モデル):
//   Publish後のWorldの型レベル不変性を保証する builder 境界 RAII wrapper。
//
// 背景:
//   当初は Coordinator テンプレートの World パラメータとして使用することを
//   想定していたが、C++ の operator-> 制約（生ポインタの built-in -> は
//   operator->() を経由しない）により、consumeWorldHandle が返す
//   const FrozenRuntimeWorld* に対して world->field がコンパイルエラーになる。
//   そのため Coordinator の World パラメータとしては使用しない。
//
// 現在の役割 (二段階モデル):
//   1. Builder 境界の RAII wrapper（aligned_unique_ptr<RuntimeState> 保持）
//   2. デストラクタで unseal() を呼び出す
//   3. const access のみ提供（コンパイル時不変性）
//   4. Coordinator は RuntimeState を直接使用（FrozenRuntimeWorld 非依存）
//
// Retire 時の unseal は bridge.retireRuntimePublishWorldNonRt が担当:
//   AudioEngine.h 内で ptr->unseal() を呼び出し → aligned_free
//   これにより FrozenRuntimeWorld が行うべき安全性を代替保証する。
//
// 将来の拡張:
//   Builder → Publish 境界で FrozenRuntimeWorld を戻り値とすることで、
//   「凍結済み」を型レベルで表明可能（段階的導入）


class FrozenRuntimeWorld {
public:
    // ★ 構築: freeze() 済みの RuntimeState を受け取る
    //   aligned_unique_ptr → state_ へムーブ
    //   .cpp で定義（完全型が必要）
    explicit FrozenRuntimeWorld(aligned_unique_ptr<RuntimeState> state) noexcept;

    // ★ const access のみ提供 — 非constアクセスはコンパイルエラー
    [[nodiscard]] const RuntimeState& get() const noexcept { return *state_; }
    [[nodiscard]] const RuntimeState& operator*() const noexcept { return *state_; }
    [[nodiscard]] const RuntimeState* operator->() const noexcept { return state_.get(); }
    [[nodiscard]] const RuntimeState* getRaw() const noexcept { return state_.get(); }

    // ★ 所有権移譲: Builder→Coordinator 間の publish 経路で使用
    //   FrozenRuntimeWorld の所有権を放棄し、内部の RuntimeState* を返す。
    //   呼出後はこのオブジェクトの state_ は nullptr になり、デストラクタは
    //   unseal を実行しない（呼出側が責任を持つ）。
    //   Coordinator の retire 経路 (bridge.retireRuntimePublishWorldNonRt) が
    //   ptr->unseal() → aligned_free(ptr) を実行することを前提とする。
    [[nodiscard]] RuntimeState* releaseState() noexcept
    {
        return state_.release();
    }

    // ★ ムーブのみ許可（unique_ptr 所有権モデル準拠）
    FrozenRuntimeWorld(FrozenRuntimeWorld&&) noexcept = default;
    FrozenRuntimeWorld& operator=(FrozenRuntimeWorld&&) noexcept = default;

    // ★ コピー禁止（unique_ptr 所有権のため）
    FrozenRuntimeWorld(const FrozenRuntimeWorld&) = delete;
    FrozenRuntimeWorld& operator=(const FrozenRuntimeWorld&) = delete;

    // ★ デストラクタ: Retire時に unseal → 解放（.cpp で定義）
    ~FrozenRuntimeWorld();

    // ★ SealedObject プロトコル互換: publishWorld() が要求するメソッド
    //    RuntimeState は既に freeze() 済みだが、Coordinator の publish パスが
    //    sealRecursively() を呼ぶため、無害なデリゲートを提供する。
    //    .cpp で定義（完全型が必要）
    void sealRecursively() noexcept;

    [[nodiscard]] bool isValid() const noexcept { return state_ != nullptr; }

private:
    // ★ aligned_unique_ptr<RuntimeState> で保持（非constでも外部には const access のみ提供）
    //   RuntimeState は aligned_malloc (64-byte align / MKL) で確保されるため、
    //   正しいデリーター（~T() + mkl_free）が必要
    aligned_unique_ptr<RuntimeState> state_;
};

} // namespace convo
