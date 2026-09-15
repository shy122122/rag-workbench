class ApiKeyMissing(RuntimeError):
    """百炼 API Key 未配置。"""


class ModelCallError(RuntimeError):
    """调用百炼接口失败。"""


class DimMismatch(RuntimeError):
    """查询向量维度与库中向量维度不一致（通常是换了嵌入模型但没重建库）。"""


class StoreCorrupt(RuntimeError):
    """向量库载入失败，为避免覆盖掉还能抢救的原文件，已禁止写入。"""
