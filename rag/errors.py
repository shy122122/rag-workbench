class ApiKeyMissing(RuntimeError):
    """百炼 API Key 未配置。"""


class ModelCallError(RuntimeError):
    """调用百炼接口失败。"""
