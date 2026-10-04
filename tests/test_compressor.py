"""
游戏内资源压缩与解压系统 - 验收测试
运行方式：python -m pytest tests/test_compressor.py -q
共 15 个测试用例
"""
import io
import json
import os
import struct
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Python 3.14+ 的 stdlib 新增了 compression 包，pytest 启动时可能已将其导入
# sys.modules，需先让位，确保导入的是本仓库的 compression 包。
for _mod in [m for m in list(sys.modules)
             if m == "compression" or m.startswith("compression.")]:
    del sys.modules[_mod]

from compression.compressor import (
    HEADER_CAPACITY,
    MAGIC,
    ChecksumError,
    ResourceCompressor,
)


def make_resource_data(size=2 * 1024 * 1024):
    """模拟游戏资源：大量重复的贴图/顶点描述文本。"""
    block = (
        b"texture diffuse_map normal_map specular_map vertex_buffer "
        b"index_buffer material shader uniform sampler2D mesh_instance "
        b"position normal tangent uv0 uv1 blend_weight skeleton_joint "
    ) * 64
    repeats = size // len(block) + 1
    return (block * repeats)[:size]


class FailingReader(io.BytesIO):
    """读取到 limit 字节后模拟网络中断。"""

    def __init__(self, data, limit):
        super().__init__(data)
        self.limit = limit
        self._read_so_far = 0

    def read(self, size=-1):
        if self._read_so_far >= self.limit:
            raise IOError("模拟下载中断")
        chunk = super().read(size)
        self._read_so_far += len(chunk)
        return chunk


class TrackingReader(io.BytesIO):
    """记录单次读取的最大字节数。"""

    def __init__(self, data):
        super().__init__(data)
        self.max_read = 0

    def read(self, size=-1):
        self.max_read = max(self.max_read, size if size >= 0 else 0)
        return super().read(size)


class TrackingWriter(io.BytesIO):
    """记录单次写入的最大字节数。"""

    def __init__(self):
        super().__init__()
        self.max_write = 0

    def write(self, data):
        self.max_write = max(self.max_write, len(data))
        return super().write(data)


class DribbleReader:
    """模拟边下载边解压：每次只返回少量字节，并记录是否已被读完。"""

    def __init__(self, data, piece=997):
        self.data = data
        self.piece = piece
        self.pos = 0
        self.exhausted_before_write = None

    def read(self, size=-1):
        if self.pos >= len(self.data):
            return b""
        size = self.piece if size < 0 else min(size, self.piece)
        chunk = self.data[self.pos:self.pos + size]
        self.pos += len(chunk)
        return chunk

    @property
    def exhausted(self):
        return self.pos >= len(self.data)


class TestResourceCompressor:
    def setup_method(self):
        self.system = ResourceCompressor()

    # 1. 压缩率：高效算法 + 字典，重复资源数据大幅缩小
    def test_case_01(self):
        data = make_resource_data()
        compressed = self.system.compress(data)
        assert len(compressed) < len(data) * 0.5, \
            f"压缩率过低: {len(data)} -> {len(compressed)}"

    # 2. 压缩/解压无损往返
    def test_case_02(self):
        for data in (b"", b"a", os.urandom(4096), make_resource_data(300000)):
            assert self.system.decompress(self.system.compress(data)) == data

    # 3. 多线程压缩：多分块并行且结果正确
    def test_case_03(self):
        system = ResourceCompressor({"num_threads": 4, "chunk_size": 64 * 1024})
        data = make_resource_data(64 * 1024 * 8)
        compressed = system.compress(data)
        header = json.loads(
            compressed[8:8 + HEADER_CAPACITY].rstrip(b"\x00").decode())
        assert system.num_threads == 4
        assert header["num_chunks"] == 8  # 分块并行压缩
        assert system.decompress(compressed) == data

    # 4. 字典持久化：保存到磁盘，新实例自动加载
    def test_case_04(self, tmp_path):
        dict_path = str(tmp_path / "dict.json")
        samples = [make_resource_data(50000)]
        builder = ResourceCompressor({"dict_path": dict_path})
        words = builder.build_dictionary(samples)
        assert os.path.exists(dict_path)
        loaded = ResourceCompressor({"dict_path": dict_path})
        assert loaded.dictionary_words == words

    # 5. 字典优化：高频常用词必须进入字典
    def test_case_05(self):
        samples = [make_resource_data(80000)]
        words = self.system.build_dictionary(samples)
        assert b"texture" in words
        assert b"vertex_buffer" in words
        assert b"material" in words

    # 6. 字典编码实际生效：编码后更小且可无损解码
    def test_case_06(self):
        self.system.build_dictionary([make_resource_data(50000)])
        data = make_resource_data(20000)
        encoded = self.system.dictionary_encoding(data)
        assert len(encoded) < len(data)
        assert self.system.dictionary_decoding(encoded) == data

    # 7. 字典提升压缩率：带字典压缩不大于裸压缩
    def test_case_07(self):
        data = make_resource_data(200000)
        plain = ResourceCompressor()
        plain_size = len(plain.compress(data))
        with_dict = ResourceCompressor()
        with_dict.build_dictionary([make_resource_data(100000)])
        dict_size = len(with_dict.compress(data))
        assert dict_size <= plain_size
        assert with_dict.decompress(with_dict.compress(data)) == data

    # 8. 校验和实际生效：篡改压缩数据必须被发现
    def test_case_08(self):
        data = make_resource_data(100000)
        blob = bytearray(self.system.compress(data))
        blob[-5] ^= 0xFF  # 篡改负载
        with pytest.raises(ChecksumError):
            self.system.decompress(bytes(blob))

    # 9. 校验和覆盖全部数据：篡改末尾分块也能检出
    def test_case_09(self):
        system = ResourceCompressor({"chunk_size": 32 * 1024})
        data = make_resource_data(32 * 1024 * 4)
        blob = bytearray(system.compress(data))
        blob[-1] ^= 0x01  # 最后一个分块的最后一字节
        with pytest.raises(ChecksumError):
            system.decompress(bytes(blob))

    # 10. 校验和带密钥：错误密钥无法通过验证（防篡改）
    def test_case_10(self):
        data = make_resource_data(50000)
        blob = self.system.compress(data)
        attacker = ResourceCompressor({"checksum_key": b"wrong-key"})
        with pytest.raises(ChecksumError):
            attacker.decompress(blob)
        assert self.system.checksum_verification(
            data, self.system._checksum([self.system._chunk_hmac(data)]))

    # 11. 解压进度：回调递增且最终到达总量
    def test_case_11(self):
        system = ResourceCompressor({"chunk_size": 16 * 1024})
        data = make_resource_data(16 * 1024 * 6)
        blob = system.compress(data)
        calls = []
        result = system.decompress(
            blob, progress_callback=lambda done, total: calls.append((done, total)))
        assert result == data
        assert len(calls) >= 2
        assert all(total == len(data) for _, total in calls)
        dones = sorted(done for done, _ in calls)
        assert dones[-1] == len(data)

    # 12. 流式处理：compress_stream/decompress_stream 往返
    def test_case_12(self):
        data = make_resource_data(150000)
        packed = io.BytesIO()
        self.system.compress_stream(io.BytesIO(data), packed)
        packed.seek(0)
        out = io.BytesIO()
        written = self.system.decompress_stream(packed, out)
        assert written == len(data)
        assert out.getvalue() == data

    # 13. 流式内存控制：单次读写不超过分块大小
    def test_case_13(self):
        chunk = 32 * 1024
        system = ResourceCompressor({"chunk_size": chunk})
        data = make_resource_data(chunk * 5)
        reader = TrackingReader(data)
        writer = TrackingWriter()
        system.compress_stream(reader, writer)
        assert reader.max_read <= chunk
        overhead = chunk * 0.02 + 64  # zlib 膨胀上限 + 帧头
        assert writer.max_write <= max(chunk + overhead,
                                       8 + HEADER_CAPACITY)

    # 14. 断点续传：中断后从检查点恢复，产物与一次性压缩一致
    def test_case_14(self, tmp_path):
        data = make_resource_data(200000)
        checkpoint = str(tmp_path / "checkpoint.json")

        reference = io.BytesIO()
        self.system.compress_stream(io.BytesIO(data), reference)

        partial = io.BytesIO()
        with pytest.raises(IOError):
            self.system.compress_stream(
                FailingReader(data, limit=len(data) // 2),
                partial, checkpoint_path=checkpoint)
        assert os.path.exists(checkpoint)

        resumed = io.BytesIO(partial.getvalue())
        self.system.compress_stream(
            io.BytesIO(data), resumed,
            checkpoint_path=checkpoint, resume=True)
        assert resumed.getvalue() == reference.getvalue()

        resumed.seek(0)
        assert self.system.decompress_stream(resumed, io.BytesIO()) or True
        resumed.seek(0)
        out = io.BytesIO()
        self.system.decompress_stream(resumed, out)
        assert out.getvalue() == data

    # 15. 流式解压：不必等全部下载完，边下边解压
    def test_case_15(self):
        system = ResourceCompressor({"chunk_size": 16 * 1024})
        # 混合不可压缩与重复数据，使压缩包远大于一个分块
        data = (make_resource_data(8000) + os.urandom(8000)) * 8
        packed = io.BytesIO()
        system.compress_stream(io.BytesIO(data), packed)
        dribble = DribbleReader(packed.getvalue(), piece=513)
        first_write_pos = {}

        class ProbeWriter(io.BytesIO):
            def write(self, chunk):
                first_write_pos.setdefault("exhausted", dribble.exhausted)
                return super().write(chunk)

        out = ProbeWriter()
        written = system.decompress_stream(dribble, out)
        # 第一块解压输出时，流尚未读完 -> 真正的边下边解压
        assert first_write_pos["exhausted"] is False
        assert written == len(data)
        assert out.getvalue() == data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
