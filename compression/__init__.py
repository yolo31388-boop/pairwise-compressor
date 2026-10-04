"""游戏资源压缩包。

兼容 Python 3.14+：3.14 在标准库新增了 `compression` 包（lzma/bz2/zipfile
等依赖 `compression._common`）。当本目录位于 sys.path 前列时，本包会遮蔽
stdlib 版本，导致标准库模块导入失败。这里把 stdlib 的 compression 目录追加
到 __path__，使 `compression._common` 等子模块仍可解析，两个包共存。
"""
import os as _os
import sysconfig as _sysconfig

_stdlib_compression = _os.path.join(_sysconfig.get_path("stdlib"), "compression")
if _os.path.isdir(_stdlib_compression) and _stdlib_compression not in __path__:
    __path__.append(_stdlib_compression)
