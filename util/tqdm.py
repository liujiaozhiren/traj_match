# tqdm_wrapper.py
from typing import Iterable, Optional

try:
    # 兼容 notebook/终端的自适应 tqdm
    from tqdm.auto import tqdm as _tqdm
except Exception:
    from tqdm import tqdm as _tqdm


class _NullTqdm:
    """一个什么都不做的 tqdm 替身，API 对齐常用方法。"""
    def __init__(self, iterable: Optional[Iterable] = None, total: Optional[int] = None, **kwargs):
        self.iterable = iterable
        self.total = total if total is not None else (len(iterable) if hasattr(iterable, "__len__") else None)
        self.n = 0  # 与 tqdm 接口对齐

    # 既支持 with 也支持直接 for
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False  # 不吞异常

    def __iter__(self):
        if self.iterable is None:
            return iter(())
        for item in self.iterable:
            yield item

    # 常用 no-op 方法
    def update(self, n: int = 1):
        self.n += n

    def set_postfix(self, *args, **kwargs):
        pass

    def set_description(self, *args, **kwargs):
        pass

    def close(self):
        pass


def tqdmu(iterable: Optional[Iterable] = None, *args, **kwargs):
    """
    包装 tqdm：
    - enable=None: 走 cfg.tqdm
    - enable=True: 强制开启
    - enable=False: 强制关闭
    用法与 tqdm 一致：with tqdmu(loader, desc="...") as tq: ...
    """
    # 延迟导入 cfg，避免循环依赖
    import cfg
    use_bar = cfg.tqdm
    if use_bar:
        return _tqdm(iterable, *args, **kwargs)
    else:
        return _NullTqdm(iterable, *args, **kwargs)