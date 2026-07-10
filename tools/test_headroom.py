#!/usr/bin/env python3
"""Headroom 機能テストスクリプト"""

import headroom
import json

print("=" * 60)
print("Headroom v0.26.0 機能テスト")
print("=" * 60)

# 1. Token counting via headroom utility
print("\n1) Token Counting")
text = 'Hello world, this is a test of headroom token counting on Windows system.'
try:
    tc = headroom.count_tokens_text(text, token_counter="claude")
    print(f"   Text: '{text}'")
    print(f"   Tokens: {tc}")
except Exception as e:
    print(f"   count_tokens_text skipped: {e}")
    # Fallback: use tiktoken directly
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)
    print(f"   Text: '{text}'")
    print(f"   Tokens (tiktoken): {len(tokens)}")

# 2. Compress messages (library mode)
print("\n2) Message Compression")
msgs = [{'role': 'user', 'content': 'This is a long test message to verify headroom compression functionality. ' * 20}]
result = headroom.compress(msgs)
print(f"   Compressed: {len(result.messages)} message(s)")
cm = result.messages[0]
orig_len = len(msgs[0]['content'])
comp_len = len(cm['content'])
reduction = 100 - (comp_len / orig_len * 100)
print(f"   Original: {orig_len} chars")
print(f"   Compressed: {comp_len} chars")
print(f"   Reduction: {reduction:.1f}%")

# 3. SmartCrusher (JSON compression) - 最も効果的な機能
print("\n3) SmartCrusher (JSON圧縮)")
sc = headroom.SmartCrusher(headroom.SmartCrusherConfig())
big_json = json.dumps({
    'users': [{'id': i, 'name': f'user_{i}', 'email': f'user{i}@test.com'}
              for i in range(100)]
})
crush_result = sc.crush(big_json)
sc_reduction = 100 - (len(crush_result.compressed) / len(big_json) * 100)
print(f"   Original: {len(big_json)} chars")
print(f"   Crushed: {len(crush_result.compressed)} chars")
print(f"   Reduction: {sc_reduction:.1f}%")
print(f"   Strategy: {crush_result.strategy}")

# 4. Pipeline
print("\n4) TransformPipeline")
try:
    pipe = headroom.TransformPipeline()
    pipe_result = pipe.run(msgs)
    print(f"   Result type: {type(pipe_result).__name__}")
except Exception as e:
    print(f"   Pipeline (expected with defaults): {e}")

# 5. Memory operations
print("\n5) Memory System")
try:
    mem = headroom.Memory(headroom.MemoryConfig())
    print(f"   Memory initialized: {type(mem).__name__}")
except Exception as e:
    print(f"   Memory init (expected without proxy): {e}")

# 6. Compression with larger content (try with CompressConfig)
print("\n6) Large Content Compression")
large_content = "\n".join([f"Line {i}: This is a simulated log line for testing compression efficiency with real-world data patterns." for i in range(50)])
large_msgs = [{'role': 'user', 'content': large_content}]
try:
    # Try with explicit CompressConfig
    config = headroom.CompressConfig()
    large_result = headroom.compress(large_msgs, config=config)
    large_orig = len(large_msgs[0]['content'])
    large_comp = len(large_result.messages[0]['content'])
    large_red = 100 - (large_comp / large_orig * 100)
    print(f"   Original: {large_orig} chars ({len(large_content.split(chr(10)))} lines)")
    print(f"   Compressed: {large_comp} chars")
    print(f"   Reduction: {large_red:.1f}%")
except Exception as e:
    print(f"   Compression: {e}")

print("\n" + "=" * 60)
print("全テスト完了")
print("=" * 60)
