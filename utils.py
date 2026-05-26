"""公用工具：字体配置 + 常用辅助函数"""
import os
import matplotlib
# 仅在没有显示器（运行 .py 脚本 + 保存 png）时切到 Agg backend
# Jupyter 会自己注入 inline backend，这里不要强制设置
if not os.environ.get('DISPLAY') and not os.environ.get('MPLBACKEND'):
    # 进一步判断：是否在 Jupyter 里（有 IPython 实例就别动 backend）
    try:
        from IPython import get_ipython
        if get_ipython() is None:
            matplotlib.use('Agg')
    except ImportError:
        matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

_FONT_CANDIDATES = [
    '/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc',
    '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
    '/System/Library/Fonts/PingFang.ttc',
]

def get_zh_font():
    for p in _FONT_CANDIDATES:
        if os.path.exists(p):
            return FontProperties(fname=p)
    return None

zh_font = get_zh_font()


def zh(ax_method, *args, **kwargs):
    """给matplotlib文字加中文字体的便捷调用"""
    if zh_font is not None:
        kwargs.setdefault('fontproperties', zh_font)
    return ax_method(*args, **kwargs)
