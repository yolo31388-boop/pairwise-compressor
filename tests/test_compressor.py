"""
游戏内资源压缩与解压系统 - 验收测试
运行方式：python -m pytest tests/test_compressor.py -q
共 15 个测试用例
"""
import hashlib
import itertools
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import compression.compressor as compressor_mod
from compression.compressor import ResourceCompressor


class TestResourceCompressor:
    def setup_method(self):
        self.system = ResourceCompressor({"num_threads": 2, "chunk_size": 64 * 1024})

    # 1. 压缩/解压往返一致
    def test_case_01(self):
        data = b"".join(bytes([i % 256]) * 17 + b"game-texture" for i in range(4000))
        compressed = self.system.compress(data)
        assert self.system.decompress(compressed) == data

    # 2. 压缩算法高效：高度重复数据压缩率远优于简单 RLE
    def test_case_02(self):
        data = b"player_position_x,player_position_y,texture_sheet_01" * 20000
        compressed = self.system.compress(data)
        assert len(compressed) < len(data) * 0.2

    # 3. 压缩不膨胀：不可压缩数据回退存储模式
    def test_case_03(self):
        data = hashlib.sha256(b"random").digest() * 4096
        compressed = self.system.compress(data)
        assert len(compressed) <= len(data) + 4096
        assert self.system.decompress(compressed) == data

    # 4. 压缩使用多线程
    def test_case_04(self):
        real_executor = compressor_mod.ThreadPoolExecutor
        instances = []

        class RecordingExecutor(real_executor):
            def __init__(self, *args, **kwargs):
                instances.append(kwargs.get("max_workers") or (args[0] if args else None))
                super().__init__(*args, **kwargs)

        compressor_mod.ThreadPoolExecutor = RecordingExecutor
        try:
            data = b"abcdefgh" * 100000
            result = self.system.decompress(self.system.compress(data))
        finally:
            compressor_mod.ThreadPoolExecutor = real_executor
        assert result == data
        assert len(instances) >= 1
        assert instances[0] == 2

    # 5. 字典编码实际生效：训练字典后同类数据压缩率更高
    def test_case_05(self):
        samples = [b"ui_button_press" * 200, b"ui_button_hover" * 200]
        self.system.dictionary_encoding(samples)
        new_asset = b"ui_button_press ui_button_hover"
        plain = ResourceCompressor()
        with_dict = self.system.compress(new_asset)
        without_dict = plain.compress(new_asset)
        assert len(with_dict) < len(without_dict)
        assert self.system.decompress(with_dict) == new_asset

    # 6. 字典经过优化：高频常用词进入字典
    def test_case_06(self):
        samples = [b"inventory_slot_01 " * 500, b"inventory_slot_02 " * 500]
        dictionary = self.system.dictionary_encoding(samples)
        assert dictionary
        assert b"inventory_slot_" in dictionary

    # 7. 字典持久化：新实例无需训练即可加载同一字典
    def test_case_07(self, tmp_path):
        dict_path = str(tmp_path / "dict.bin")
        trainer = ResourceCompressor({"dict_path": dict_path})
        dictionary = trainer.dictionary_encoding([b"quest_reward_gold" * 50])
        assert os.path.exists(dict_path)
        loaded = ResourceCompressor({"dict_path": dict_path}).dictionary_encoding()
        assert loaded == dictionary

    # 8. 解压校验实际生效：损坏数据被拒绝
    def test_case_08(self):
        compressed = bytearray(self.system.compress(b"save_data" * 1000))
        middle = len(compressed) // 2
        compressed[middle] ^= 0xFF
        with pytest.raises(ValueError):
            self.system.decompress(bytes(compressed))

    # 9. 校验和加密：HMAC 密钥不匹配时校验失败
    def test_case_09(self):
        data = b"checksum_protected_payload"
        system_b = ResourceCompressor({"checksum_key": "attacker-key"})
        checksum = self.system.checksum_verification(data)
        assert self.system.checksum_verification(data, checksum) is True
        assert system_b.checksum_verification(data, checksum) is False
        compressed = self.system.compress(data)
        with pytest.raises(ValueError):
            system_b.decompress(compressed)

    # 10. 校验全覆盖：头部、负载中部、负载尾部、校验和本身被篡改都能发现
    def test_case_10(self):
        compressed = bytearray(self.system.compress(b"coverage_" * 5000))
        positions = [10, len(compressed) // 2, len(compressed) - 40, len(compressed) - 5]
        for position in positions:
            tampered = bytearray(compressed)
            tampered[position] ^= 0x01
            with pytest.raises(ValueError):
                self.system.decompress(bytes(tampered))

    # 11. 解压有进度：回调单调递增且最终到达总量
    def test_case_11(self):
        data = b"progress_tracking" * 5000
        progress = []
        self.system.decompress(self.system.compress(data), progress_callback=lambda d, t: progress.append((d, t)))
        assert len(progress) >= 2
        dones = [d for d, _ in progress]
        assert dones == sorted(dones)
        assert progress[0][1] == len(data)
        assert progress[-1] == (len(data), len(data))

    # 12. 流式往返：逐块压缩、逐帧解压结果一致
    def test_case_12(self):
        chunks = [b"stream-chunk-%04d" % i + os.urandom(8) for i in range(50)]
        frames = list(self.system.streaming_compress(chunks))
        output = b"".join(self.system.streaming_decompress(iter(frames)))
        assert output == b"".join(chunks)

    # 13. 真正流式：第一帧到达即可产出，无需全部下载
    def test_case_13(self):
        chunks = [b"incremental_%04d" % i for i in range(30)]
        all_frames = list(self.system.streaming_compress(chunks))
        consumed = []

        def frame_source():
            for frame in all_frames:
                consumed.append(True)
                yield frame

        decoder = self.system.streaming_decompress(frame_source())
        first_output = next(iter(decoder))
        assert first_output == chunks[0]
        assert len(consumed) == 1
        assert b"".join(itertools.chain([first_output], decoder)) == b"".join(chunks)

    # 14. 流式内存控制：每帧不超过 chunk_size 上限
    def test_case_14(self):
        system = ResourceCompressor({"chunk_size": 4096})
        data = b"memory_bounded" * 5000
        frames = list(system.streaming_compress([data[i:i + 4096] for i in range(0, len(data), 4096)]))
        for frame in frames:
            assert len(frame) <= 4096 + 256

    # 15. 流式断点续传：中断后继续，数据完整且不重复处理
    def test_case_15(self, tmp_path):
        state_path = str(tmp_path / "stream_state.json")
        chunks = [b"resume_chunk_%04d" % i for i in range(40)]
        source = b"".join(chunks)

        first_session = self.system.streaming_compress(iter(chunks), state_path=state_path)
        frames = list(itertools.islice(first_session, 15))
        import json
        assert json.load(open(state_path))["next_index"] == 15

        resumed = list(self.system.streaming_compress(iter(chunks), state_path=state_path))
        assert len(resumed) == 25  # 已完成的 15 块没有重做
        frames += resumed

        output = b"".join(self.system.streaming_decompress(iter(frames)))
        assert output == source

        # 断点续传重发的重复帧会被自动去重，输出依然完整
        dedup_state = str(tmp_path / "dedup_state.json")
        duplicated = frames[:10] + frames  # 前 10 帧重发
        dedup_output = b"".join(
            self.system.streaming_decompress(iter(duplicated), state_path=dedup_state)
        )
        assert dedup_output == source


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
