# -*- coding: utf-8 -*-
"""
依赖管理 - 极简版
基础功能零重型依赖，语义搜索可选
"""
import importlib
import subprocess
import sys
from typing import Tuple, Dict

OPTIONAL_PACKAGES = {
    "chromadb": "chromadb",
    "sentence_transformers": "sentence-transformers",
    "torch": "torch",
}


def check_package(name: str) -> bool:
    try:
        importlib.import_module(name)
        return True
    except ImportError:
        return False


def install_package(pip_name: str, mirror: str = "") -> bool:
    cmd = [sys.executable, "-m", "pip", "install", "--quiet", "--disable-pip-version-check"]
    if mirror:
        host = mirror.split("//")[1].split("/")[0]
        cmd.extend(["-i", mirror, "--trusted-host", host])
    cmd.append(pip_name)
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300,
                          env={**dict(__import__("os").environ), "PIP_NO_INPUT": "1"})
        return r.returncode == 0
    except Exception:
        return False


def ensure_optional(logger_fn=None) -> Dict[str, bool]:
    """检查可选依赖状态"""
    if logger_fn is None:
        try:
            from astrbot.api import logger
            logger_fn = logger.info
        except Exception:
            logger_fn = print
    result = {}
    for imp_name, pip_name in OPTIONAL_PACKAGES.items():
        avail = check_package(imp_name)
        result[imp_name] = avail
        if not avail:
            logger_fn(f"[MemoryWeight] 可选依赖 {pip_name} 未安装，语义搜索不可用")
    return result


def try_install_missing(logger_fn=None) -> Dict[str, bool]:
    """尝试安装缺失的可选依赖"""
    if logger_fn is None:
        try:
            from astrbot.api import logger
            logger_fn = logger.info
        except Exception:
            logger_fn = print
    mirrors = [
        "https://mirrors.aliyun.com/pypi/simple/",
        "https://pypi.tuna.tsinghua.edu.cn/simple/",
        "https://pypi.org/simple",
    ]
    result = {}
    for imp_name, pip_name in OPTIONAL_PACKAGES.items():
        if check_package(imp_name):
            result[imp_name] = True
            continue
        installed = False
        for m in mirrors:
            logger_fn(f"[MemoryWeight] 尝试安装 {pip_name} (源: {m})...")
            if install_package(pip_name, m):
                installed = True
                logger_fn(f"[MemoryWeight] {pip_name} 安装成功")
                break
        result[imp_name] = installed
    return result
