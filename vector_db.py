# -*- coding: utf-8 -*-
"""
向量数据库层 - ChromaDB 语义搜索（可选功能）
如果 chromadb/sentence-transformers 未安装，所有方法返回空结果
"""
import os
import re
from pathlib import Path
from typing import List, Dict, Optional
from astrbot.api import logger

_HAS_CHROMADB = False
try:
    import chromadb
    _HAS_CHROMADB = True
except ImportError:
    pass


class VectorDB:
    """向量记忆数据库（可选）"""

    def __init__(self, db_path, model_name="paraphrase-multilingual-MiniLM-L12-v2",
                 model_cache_dir=None, hf_endpoint="", trust_remote_code=False, offline_mode=False):
        if not _HAS_CHROMADB:
            raise RuntimeError("chromadb 未安装，语义搜索不可用")
        self.db_path = db_path
        self.offline_mode = offline_mode
        if self.offline_mode:
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            os.environ["HF_HUB_OFFLINE"] = "1"
        else:
            os.environ.pop("TRANSFORMERS_OFFLINE", None)
            os.environ.pop("HF_HUB_OFFLINE", None)
            if hf_endpoint:
                os.environ["HF_ENDPOINT"] = hf_endpoint
        self.client = chromadb.PersistentClient(
            path=str(db_path),
            settings=chromadb.Settings(anonymized_telemetry=False, allow_reset=True)
        )
        self.collection = self.client.get_or_create_collection(name="memories")
        self.model = None
        self.model_name = model_name
        self.model_cache_dir = model_cache_dir
        self.hf_endpoint = hf_endpoint
        self.trust_remote_code = trust_remote_code

    def _ensure_model(self):
        if self.model is None:
            self._load_model()

    def _load_model(self):
        from sentence_transformers import SentenceTransformer
        import torch
        search_roots = []
        for env_var in ["SENTENCE_TRANSFORMERS_HOME", "HF_HOME", "XDG_CACHE_HOME"]:
            val = os.environ.get(env_var)
            if val:
                p = Path(val)
                if env_var == "HF_HOME":
                    p = p / "sentence_transformers"
                if env_var == "XDG_CACHE_HOME":
                    p = p / "torch" / "sentence_transformers"
                search_roots.append(p)
        try:
            default_torch_home = torch.hub._get_torch_home()
            search_roots.append(Path(default_torch_home) / "sentence_transformers")
        except Exception:
            pass
        search_roots.append(Path.home() / ".cache" / "torch" / "sentence_transformers")
        search_roots.append(Path.home() / ".cache" / "huggingface" / "sentence_transformers")
        plugin_cache_dir = Path(self.model_cache_dir).expanduser().resolve() if self.model_cache_dir else Path(self.db_path).parent / "MemoryWeight_ModelCache"
        search_roots.append(plugin_cache_dir)

        def find_local_path(roots, name):
            short_name = name.split("/")[-1]
            for root in roots:
                if not root.exists():
                    continue
                for sn in [name, name.replace("/", "_"), short_name]:
                    p = root / sn
                    if p.exists() and (p / "config.json").exists():
                        return str(p.resolve())
                try:
                    for p in root.iterdir():
                        if p.is_dir() and short_name in p.name and (p / "config.json").exists():
                            return str(p.resolve())
                except Exception:
                    continue
            return None

        found = find_local_path(search_roots, self.model_name)
        if found:
            try:
                self.model = SentenceTransformer(found, trust_remote_code=self.trust_remote_code, local_files_only=True)
                logger.info(f"[MemoryWeight] 本地加载模型: {found}")
                return
            except Exception:
                pass
        if self.offline_mode:
            raise RuntimeError(f"离线模式未找到模型 {self.model_name}")
        os.environ["HF_HOME"] = str(plugin_cache_dir)
        if self.hf_endpoint:
            os.environ["HF_ENDPOINT"] = self.hf_endpoint
        self.model = SentenceTransformer(self.model_name, trust_remote_code=self.trust_remote_code, cache_folder=str(plugin_cache_dir))
        logger.info(f"[MemoryWeight] 下载模型: {self.model_name}")

    def get_embeddings(self, texts):
        if not texts:
            return []
        self._ensure_model()
        return self.model.encode(texts, normalize_embeddings=True).tolist()

    def add_memories(self, memories):
        if not memories:
            return
        self._ensure_model()
        ids = [m["memory_id"] for m in memories]
        docs = [m["content"] for m in memories]
        embeds = self.model.encode(docs, normalize_embeddings=True).tolist()
        self.collection.upsert(ids=ids, embeddings=embeds, documents=docs)

    def search_memories(self, query, top_n=10):
        self._ensure_model()
        qe = self.model.encode([query], normalize_embeddings=True).tolist()
        results = self.collection.query(query_embeddings=qe, n_results=top_n)
        output = []
        if results["ids"] and results["distances"]:
            for eid, dist in zip(results["ids"][0], results["distances"][0]):
                relevance = max(0, 1 - dist / 2) * 100
                output.append({"memory_id": eid, "relevance": round(relevance, 1)})
        return output

    def clear_all(self):
        try:
            self.client.delete_collection(name="memories")
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(name="memories")
