# -*- coding: utf-8 -*-
"""
astrbot_plugin_memory_weight - 模拟人类记忆风格的记忆权重插件

核心功能：
- 记忆存储与权重管理
- Ebbinghaus遗忘曲线衰减
- 间隔重复强化
- 语义向量检索
- 记忆巩固（短期→长期）
- 情感标记与加权
- 回收站机制
- 自然语言触发
"""
import asyncio
import json
import os
import uuid
import re
import math
import random
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any

from astrbot.api.event import filter, AstrMessageEvent, MessageEventResult
from astrbot.api.star import Context, Star, register, StarTools
from astrbot.api import logger

from .models import Memory, MemoryNode, MemoryRelation, MemoryType, EmotionTag
from .database import MemoryDB
from .memory_decay import MemoryDecayEngine
from .deps import ensure_optional, check_package


@register("memory_weight", "AutoClaw", "模拟人类记忆风格的记忆权重插件 - 支持遗忘曲线、间隔重复、情感加权、语义检索", "1.0.0")
class MemoryWeightPlugin(Star):
    def __init__(self, context: Context, config: dict = None):
        super().__init__(context)
        self.context = context
        self.config = config or {}
        
        # Check optional dependencies (vector search)
        self._has_vector_db = check_package("chromadb") and check_package("sentence_transformers")
        if not self._has_vector_db:
            logger.info("[MemoryWeight] 语义搜索不可用（缺少 chromadb/sentence-transformers），仅使用关键词搜索")
        data_dir = StarTools.get_data_dir()
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
        db_path = self.config.get("memory_db_path", "memory_weight.db")
        if not os.path.isabs(db_path):
            db_path = str(data_dir / db_path)
        self.db_path = db_path
        vector_db_path = self.config.get("vector_db_path", "MemoryWeight_VectorDB")
        if not os.path.isabs(vector_db_path):
            vector_db_path = str(data_dir / vector_db_path)
        self.vector_db_path = vector_db_path
        self.db = MemoryDB(self.db_path)
        self.decay_engine = MemoryDecayEngine(self.config)
        self.vector_db = None
        self.recall_keywords = ["记得", "记住", "之前", "以前", "上次", "回忆", "想起来", "还记得"]
        self.forget_keywords = ["忘掉", "忘记", "删除记忆", "清除记忆"]
        self.store_keywords = ["记住", "记下", "帮我记", "别忘了", "以后提醒"]
        self.auto_listen = self._to_bool(self.config.get("auto_listen", True))
        self.similarity_threshold = float(self.config.get("similarity_threshold", 0.8))
        self.strength_threshold = float(self.config.get("strength_threshold", 5.0))
        self.auto_recall_probability = float(self.config.get("auto_recall_probability", 0.2))
        self.auto_recall_threshold = float(self.config.get("auto_recall_threshold", 40.0))
        self.reinforcement_intensity = float(self.config.get("reinforcement_intensity", 1.0))
        self.max_memories_display = int(self.config.get("max_memories_display", 20))
        self.max_memories_per_request = int(self.config.get("max_memories_per_request", 5))
        self.offline_mode = self._to_bool(self.config.get("offline_mode", False))
        self.hf_endpoint = self.config.get("hf_endpoint", "https://hf-mirror.com").strip()
        self.embedding_model = self.config.get("embedding_model", "paraphrase-multilingual-MiniLM-L12-v2")
        embedding_cache_dir_rel = self.config.get("embedding_cache_dir", "MemoryWeight_ModelCache")
        if self.offline_mode:
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            os.environ["HF_HUB_OFFLINE"] = "1"
        else:
            os.environ.pop("TRANSFORMERS_OFFLINE", None)
            os.environ.pop("HF_HUB_OFFLINE", None)
            if self.hf_endpoint:
                os.environ["HF_ENDPOINT"] = self.hf_endpoint
        self.embedding_cache_dir = None
        if embedding_cache_dir_rel:
            p = Path(embedding_cache_dir_rel)
            if not p.is_absolute():
                p = data_dir / p
            self.embedding_cache_dir = str(p)
            p.mkdir(parents=True, exist_ok=True)
        logger.info("[MemoryWeight] 插件初始化完成")

    def _to_bool(self, value, default=False):
        if isinstance(value, bool): return value
        if value is None: return default
        if isinstance(value, str): return value.strip().lower() in ["1", "true", "yes", "on"]
        return bool(value)

    def _ensure_vector_db(self):
        if not self._has_vector_db:
            return False
        if self.vector_db is None:
            try:
                from .vector_db import VectorDB
                self.vector_db = VectorDB(
                    db_path=self.vector_db_path, model_name=self.embedding_model,
                    model_cache_dir=self.embedding_cache_dir, hf_endpoint=self.hf_endpoint,
                    trust_remote_code=self._to_bool(self.config.get("embedding_trust_remote_code", False)),
                    offline_mode=self.offline_mode
                )
            except Exception as e:
                logger.warning(f"[MemoryWeight] 向量数据库初始化失败: {e}")
                self._has_vector_db = False
                return False
        return True

    def _get_unified_id(self, event):
        actual_event = getattr(event, 'event', event)
        uid = getattr(actual_event, 'unified_id', None)
        if uid: return uid
        uid = getattr(actual_event, 'unified_msg_origin', None)
        if uid: return uid
        if hasattr(actual_event, 'get_unified_id'):
            try: return actual_event.get_unified_id()
            except: pass
        try:
            msg_obj = getattr(actual_event, 'message_obj', None)
            if msg_obj:
                platform = getattr(msg_obj, 'platform', '')
                sender = getattr(msg_obj, 'sender', None)
                if sender and hasattr(sender, 'user_id'):
                    return f"{platform}:{sender.user_id}"
        except: pass
        return None

    def _get_sender_nickname(self, event):
        actual_event = getattr(event, 'event', event)
        if hasattr(actual_event, 'get_sender_name'):
            try:
                name = actual_event.get_sender_name()
                if name: return name
            except: pass
        try:
            msg_obj = getattr(actual_event, 'message_obj', None)
            if msg_obj and hasattr(msg_obj, 'sender'):
                sender = msg_obj.sender
                for attr in ['nickname', 'name', 'card', 'user_id']:
                    val = getattr(sender, attr, None)
                    if val: return str(val)
        except: pass
        return "User"

    def _generate_memory_id(self):
        now = datetime.now()
        date_str = now.strftime("%Y%m%d")
        existing = self.db.get_all_memories()
        today_count = sum(1 for m in existing if m.get("created_at", "").startswith(date_str))
        return f"mem_{date_str}_{today_count + 1:03d}"

    def _detect_emotion(self, text):
        emotion_patterns = {
            "happy": (["开心", "高兴", "快乐", "幸福", "愉快", "哈哈", "嘻嘻", "太好了", "棒", "好棒", "厉害"], 6.0),
            "sad": (["难过", "伤心", "悲伤", "哭泣", "哭", "呜呜", "好伤心", "心痛"], 6.0),
            "angry": (["生气", "愤怒", "气死", "烦死", "讨厌", "可恶", "混蛋"], 7.0),
            "excited": (["兴奋", "激动", "太棒了", "哇", "天哪", "我的天", "太厉害了"], 7.0),
            "grateful": (["谢谢", "感谢", "多谢", "感恩", "谢了"], 5.0),
            "anxious": (["担心", "焦虑", "紧张", "害怕", "恐惧", "慌"], 5.0),
            "nostalgic": (["怀念", "回忆", "以前", "那时候", "想当年"], 4.0),
            "peaceful": (["平静", "安静", "宁静", "放松", "舒服", "惬意"], 3.0),
        }
        text_lower = text.lower()
        best_emotion = "neutral"
        best_intensity = 0.0
        for emotion, (keywords, base_intensity) in emotion_patterns.items():
            for kw in keywords:
                if kw in text_lower:
                    matches = sum(1 for k in keywords if k in text_lower)
                    intensity = min(base_intensity + matches * 0.5, 10.0)
                    if intensity > best_intensity:
                        best_emotion = emotion
                        best_intensity = intensity
                    break
        return best_emotion, best_intensity

    def _extract_tags(self, text):
        tags = []
        person_patterns = ["我", "你", "他", "她", "它", "我们", "你们", "他们"]
        for p in person_patterns:
            if p in text: tags.append(p)
        time_patterns = ["今天", "明天", "昨天", "下周", "上周", "这个月", "去年"]
        for t in time_patterns:
            if t in text: tags.append(t)
        return list(set(tags))[:5]

    def _is_duplicate(self, text):
        existing = self.db.get_all_memories()
        for mem in existing:
            similarity = self._text_similarity(text, mem.get("content", ""))
            if similarity > self.similarity_threshold:
                return mem
        return None

    @staticmethod
    def _text_similarity(text1, text2):
        if not text1 or not text2: return 0.0
        chars1, chars2 = set(text1), set(text2)
        if not chars1 or not chars2: return 0.0
        return len(chars1 & chars2) / max(len(chars1), len(chars2))

    def _add_or_update_memory(self, content, source="user_input", emotion=None, emotional_intensity=None, memory_type="episodic", tags=None, context=""):
        existing = self._is_duplicate(content)
        if existing:
            now = datetime.now(timezone.utc).isoformat()
            old_strength = existing.get("strength", 50.0)
            new_stability, new_strength = self.decay_engine.apply_recall_boost(
                current_stability=existing.get("stability", 1.0),
                current_strength=old_strength, recall_count=existing.get("recall_count", 0)
            )
            if not emotion:
                detected_emotion, detected_intensity = self._detect_emotion(content)
                emotion = emotion or detected_emotion
                emotional_intensity = emotional_intensity or detected_intensity
            updates = {
                "last_recalled_at": now, "recall_count": existing.get("recall_count", 0) + 1,
                "strength": new_strength, "stability": new_stability,
                "emotion": emotion or existing.get("emotion", "neutral"),
                "emotional_intensity": max(emotional_intensity or 0, existing.get("emotional_intensity", 0)),
            }
            self.db.update_memory(existing["memory_id"], updates)
            if self._ensure_vector_db():
                try: self.vector_db.add_memories([{"memory_id": existing["memory_id"], "content": content}])
                except Exception as e: logger.warning(f"[MemoryWeight] 向量更新失败: {e}")
            existing.update(updates)
            return False, existing, new_strength - old_strength
        else:
            now = datetime.now(timezone.utc).isoformat()
            if not emotion: emotion, emotional_intensity = self._detect_emotion(content)
            elif emotional_intensity is None: _, emotional_intensity = self._detect_emotion(content)
            base_strength = float(self.config.get("initial_strength", 50.0))
            if emotional_intensity and emotional_intensity > 5.0:
                base_strength = min(100.0, base_strength + (emotional_intensity - 5.0) * 2.0)
            base_stability = float(self.config.get("base_stability", 1.0))
            if emotional_intensity and emotional_intensity > 5.0:
                base_stability *= (1.0 + (emotional_intensity - 5.0) * 0.1)
            if not tags: tags = self._extract_tags(content)
            memory = {
                "memory_id": self._generate_memory_id(), "content": content,
                "memory_type": memory_type, "strength": base_strength,
                "initial_strength": base_strength, "stability": base_stability,
                "created_at": now, "last_recalled_at": now, "recall_count": 0,
                "consolidation_level": 0, "emotion": emotion,
                "emotional_intensity": emotional_intensity, "tags": tags,
                "source": source, "context": context, "related_memory_ids": []
            }
            self.db.insert_memory(memory)
            if self._ensure_vector_db():
                try: self.vector_db.add_memories([{"memory_id": memory["memory_id"], "content": content}])
                except Exception as e: logger.warning(f"[MemoryWeight] 向量添加失败: {e}")
            return True, memory, 0.0

    def _purge_weak_memories(self):
        all_memories = self.db.get_all_memories()
        purged = 0
        for mem in all_memories:
            current_strength = self.decay_engine.calculate_strength(
                initial_strength=mem.get("initial_strength", 50.0),
                stability=mem.get("stability", 1.0),
                last_recalled_at=mem.get("last_recalled_at", mem.get("created_at", "")),
                consolidation_level=mem.get("consolidation_level", 0),
                emotional_intensity=mem.get("emotional_intensity", 0.0),
                recall_count=mem.get("recall_count", 0)
            )
            if current_strength <= self.strength_threshold:
                self.db.archive_memory(mem["memory_id"], reason="strength_zero")
                purged += 1
            else:
                self.db.update_memory(mem["memory_id"], {"strength": current_strength})
        if purged > 0: logger.info(f"[MemoryWeight] 已归档 {purged} 条强度过低的记忆")
        return purged

    def _consolidate_memories(self):
        all_memories = self.db.get_all_memories()
        consolidated = 0
        for mem in all_memories:
            if self.decay_engine.should_consolidate(mem):
                new_level = min(5, mem.get("consolidation_level", 0) + 1)
                stability_boost = 1.0 + (new_level * 0.2)
                new_stability = mem.get("stability", 1.0) * stability_boost
                self.db.update_memory(mem["memory_id"], {"consolidation_level": new_level, "stability": new_stability})
                consolidated += 1
        return consolidated

    def _get_emotion_icon(self, emotion):
        icons = {"happy": "😊", "sad": "😢", "angry": "😠", "excited": "🤩", "grateful": "🙏",
                 "anxious": "😰", "nostalgic": "🥺", "peaceful": "😌", "neutral": "😐"}
        return icons.get(emotion, "💭")

    def _get_help_text(self):
        return (
            "🧠 **记忆权重插件 - 帮助**\n\n"
            "**指令列表：**\n"
            "/mem 记住 <内容>       - 添加新记忆\n"
            "/mem 列表              - 查看当前记忆\n"
            "/mem 搜索 <关键词>     - 搜索记忆\n"
            "/mem 详情 <ID>         - 查看记忆详情\n"
            "/mem 强化 <ID>         - 手动强化记忆\n"
            "/mem 删除 <ID>         - 删除指定记忆\n"
            "/mem 统计              - 查看统计信息\n"
            "/mem 最强              - 查看最强记忆\n"
            "/mem 最弱              - 查看最弱记忆\n"
            "/mem 情感 <类型>       - 按情感查看记忆\n"
            "/mem 回收站            - 查看回收站\n"
            "/mem 巩固              - 执行记忆巩固\n"
            "/mem 清理              - 清理弱记忆\n\n"
            "**情感类型：** happy, sad, angry, excited, grateful, anxious, nostalgic, peaceful\n\n"
            "**自然语言触发：**\n"
            "- 「记住...」→ 自动添加记忆\n"
            "- 「还记得...」→ 自动搜索回忆\n"
            "- 「忘掉...」→ 自动删除记忆\n\n"
            "**记忆原理：**\n"
            "基于Ebbinghaus遗忘曲线，记忆会随时间自然衰减。通过回忆可以强化记忆，高情感强度的记忆更不容易被遗忘。"
        )

    @filter.command("mem")
    async def memory_command(self, event: AstrMessageEvent):
        message_str = event.message_str.strip()
        parts = message_str.split(maxsplit=1)
        if len(parts) < 2:
            yield event.plain_result(self._get_help_text())
            return
        sub_cmd = parts[1].strip()
        if sub_cmd.startswith("记住"):
            content = sub_cmd[2:].strip()
            if not content:
                yield event.plain_result("请提供要记住的内容，例如：/mem 记住 我喜欢喝冰美式")
                return
            is_new, mem, change = self._add_or_update_memory(content, "user_command")
            if is_new:
                yield event.plain_result(f"✅ 已记住：{mem['content']}\n📊 初始强度：{mem['strength']:.1f} | 情感：{mem['emotion']}\n🆔 记忆ID：{mem['memory_id']}")
            elif change > 0:
                yield event.plain_result(f"🔄 你说的这件事我更清楚地记起来了！\n📝 {mem['content']}\n📊 当前强度：{mem['strength']:.1f} (+{change:.1f})")
            else:
                yield event.plain_result(f"💭 这件事我已经有印象了~\n📝 {mem['content']}\n📊 当前强度：{mem['strength']:.1f} | 已回忆 {mem.get('recall_count', 0)} 次")
        elif sub_cmd == "列表":
            self._purge_weak_memories()
            memories = self.db.get_all_memories()
            if not memories:
                yield event.plain_result("📭 当前没有任何记忆。")
                return
            lines = ["📋 **当前记忆列表：**\n"]
            for i, mem in enumerate(memories[:self.max_memories_display], 1):
                strength = mem.get("strength", 0)
                label = self.decay_engine.get_strength_label(strength)
                icon = self._get_emotion_icon(mem.get("emotion", "neutral"))
                lines.append(f"{i}. {icon} [{strength:.0f}] {mem['content'][:40]}...\n   └─ {label} | 回忆{mem.get('recall_count', 0)}次 | {mem['memory_type']}")
            if len(memories) > self.max_memories_display:
                lines.append(f"\n... 还有 {len(memories) - self.max_memories_display} 条记忆")
            yield event.plain_result("\n".join(lines))
        elif sub_cmd == "回收站":
            bin_items = self.db.get_recycle_bin()
            if not bin_items:
                yield event.plain_result("🗑️ 回收站为空。")
                return
            lines = ["🗑️ **回收站：**\n"]
            for i, item in enumerate(bin_items[:10], 1):
                lines.append(f"{i}. [{item.get('archived_at', '?')}] {item['content'][:40]}...\n   └─ 原始强度：{item.get('initial_strength', 0):.0f}")
            yield event.plain_result("\n".join(lines))
        elif sub_cmd.startswith("搜索"):
            query = sub_cmd[2:].strip()
            if not query:
                yield event.plain_result("请提供搜索关键词，例如：/mem 搜索 喜欢")
                return
            memories = self.db.search_memories(query)
            if not memories:
                yield event.plain_result(f"🔍 未找到与「{query}」相关的记忆。")
                return
            lines = [f"🔍 **搜索「{query}」的结果：**\n"]
            for i, mem in enumerate(memories[:10], 1):
                lines.append(f"{i}. [{mem.get('strength', 0):.0f}] {mem['content'][:50]}...")
            yield event.plain_result("\n".join(lines))
        elif sub_cmd == "统计":
            stats = self.db.count_memories()
            yield event.plain_result(f"📊 **记忆统计：**\n📦 活跃记忆：{stats['active']} 条\n🗑️ 已归档：{stats['archived']} 条\n📈 平均强度：{stats['avg_strength']:.1f}\n🧠 记忆引擎：Ebbinghaus遗忘曲线 + 间隔重复")
        elif sub_cmd.startswith("强化"):
            memory_id = sub_cmd[2:].strip()
            if not memory_id:
                yield event.plain_result("请提供记忆ID，例如：/mem 强化 mem_20260501_001")
                return
            mem = self.db.get_memory(memory_id)
            if not mem:
                yield event.plain_result(f"❌ 未找到记忆 {memory_id}")
                return
            new_stability, new_strength = self.decay_engine.apply_recall_boost(
                current_stability=mem.get("stability", 1.0),
                current_strength=mem.get("strength", 50.0),
                recall_count=mem.get("recall_count", 0)
            )
            self.db.update_memory(memory_id, {"strength": new_strength, "stability": new_stability, "last_recalled_at": datetime.now(timezone.utc).isoformat(), "recall_count": mem.get("recall_count", 0) + 1})
            yield event.plain_result(f"💪 记忆已强化！\n📝 {mem['content'][:40]}...\n📊 强度：{mem.get('strength', 0):.1f} → {new_strength:.1f}")
        elif sub_cmd == "巩固":
            count = self._consolidate_memories()
            yield event.plain_result(f"🔒 记忆巩固完成，{count} 条记忆被提升。")
        elif sub_cmd == "清理":
            purged = self._purge_weak_memories()
            yield event.plain_result(f"🧹 清理完成，{purged} 条弱记忆已归档。")
        elif sub_cmd.startswith("情感"):
            emotion = sub_cmd[2:].strip()
            if not emotion:
                yield event.plain_result("请指定情感类型，例如：/mem 情感 happy")
                return
            memories = self.db.get_memories_by_emotion(emotion)
            if not memories:
                yield event.plain_result(f"🔍 未找到情感为「{emotion}」的记忆。")
                return
            lines = [f"🎭 **情感「{emotion}」的记忆：**\n"]
            for i, mem in enumerate(memories[:10], 1):
                lines.append(f"{i}. [{mem.get('strength', 0):.0f}] {mem['content'][:40]}...")
            yield event.plain_result("\n".join(lines))
        elif sub_cmd == "最强":
            memories = self.db.get_strongest_memories(10)
            if not memories:
                yield event.plain_result("📭 没有记忆。")
                return
            lines = ["💪 **最强记忆：**\n"]
            for i, mem in enumerate(memories, 1):
                lines.append(f"{i}. [{mem.get('strength', 0):.0f}] {mem['content'][:40]}...")
            yield event.plain_result("\n".join(lines))
        elif sub_cmd == "最弱":
            memories = self.db.get_weakest_memories(10)
            if not memories:
                yield event.plain_result("📭 没有记忆。")
                return
            lines = ["⚠️ **最弱记忆（即将遗忘）：**\n"]
            for i, mem in enumerate(memories, 1):
                lines.append(f"{i}. [{mem.get('strength', 0):.0f}] {mem['content'][:40]}...")
            yield event.plain_result("\n".join(lines))
        elif sub_cmd.startswith("删除"):
            memory_id = sub_cmd[2:].strip()
            if not memory_id:
                yield event.plain_result("请提供记忆ID")
                return
            mem = self.db.get_memory(memory_id)
            if not mem:
                yield event.plain_result(f"❌ 未找到记忆 {memory_id}")
                return
            self.db.archive_memory(memory_id, reason="user_delete")
            yield event.plain_result(f"🗑️ 已删除记忆：{mem['content'][:40]}...")
        elif sub_cmd.startswith("详情"):
            memory_id = sub_cmd[2:].strip()
            if not memory_id:
                yield event.plain_result("请提供记忆ID")
                return
            mem = self.db.get_memory(memory_id)
            if not mem:
                yield event.plain_result(f"❌ 未找到记忆 {memory_id}")
                return
            strength = mem.get("strength", 0)
            label = self.decay_engine.get_strength_label(strength)
            lines = [f"📖 **记忆详情：**\n", f"🆔 ID：{mem['memory_id']}", f"📝 内容：{mem['content']}", f"📊 强度：{strength:.1f} ({label})", f"🔧 稳定性：{mem.get('stability', 1.0):.2f}", f"🔄 回忆次数：{mem.get('recall_count', 0)}", f"🔒 巩固等级：{mem.get('consolidation_level', 0)}/5", f"🎭 情感：{mem.get('emotion', 'neutral')}", f"📁 类型：{mem.get('memory_type', 'episodic')}", f"🏷️ 标签：{', '.join(mem.get('tags', []))}", f"📅 创建：{mem.get('created_at', '?')}", f"⏰ 最后回忆：{mem.get('last_recalled_at', '?')}"]
            yield event.plain_result("\n".join(lines))
        elif sub_cmd == "帮助":
            yield event.plain_result(self._get_help_text())
        else:
            yield event.plain_result("❓ 未知指令。使用 /mem 帮助 查看用法。")

    @filter.regex(r"^(记住|记下|帮我记|别忘了|以后提醒).+")
    async def natural_store_listener(self, event: AstrMessageEvent):
        message_str = event.message_str.strip()
        content = re.sub(r"^(记住|记下|帮我记|别忘了|以后提醒)\s*", "", message_str, count=1)
        if not content: return
        is_new, mem, change = self._add_or_update_memory(content, "natural_language")
        if is_new:
            yield event.plain_result(f"✅ 好的，我记住了！\n📝 {mem['content']}\n📊 强度：{mem['strength']:.1f} | 情感：{mem['emotion']}")
        elif change > 0:
            yield event.plain_result(f"🔄 嗯，这件事我印象更深了！\n📝 {mem['content']}\n📊 强度：{mem['strength']:.1f} (+{change:.1f})")

    @filter.regex(r"^(还记得|记得|之前|以前|上次|回忆|想起来).+")
    async def natural_recall_listener(self, event: AstrMessageEvent):
        message_str = event.message_str.strip()
        content = re.sub(r"^(还记得|记得|之前|以前|上次|回忆|想起来)\s*", "", message_str, count=1)
        if not content or len(content) < 2: return
        memories = self.db.search_memories(content, limit=5, min_strength=10.0)
        if not memories and self._ensure_vector_db():
            try:
                search_results = self.vector_db.search_memories(content, top_n=5)
                for result in search_results:
                    mem = self.db.get_memory(result["memory_id"])
                    if mem and mem.get("strength", 0) >= 10.0:
                        memories.append(mem)
            except Exception as e:
                logger.warning(f"[MemoryWeight] 向量搜索失败: {e}")
        if memories:
            lines = [f"💭 关于「{content}」，我记得：\n"]
            for i, mem in enumerate(memories[:3], 1):
                strength = mem.get("strength", 0)
                label = self.decay_engine.get_strength_label(strength)
                icon = self._get_emotion_icon(mem.get("emotion", "neutral"))
                lines.append(f"{i}. {icon} {mem['content'][:50]}... ({label})")
            for mem in memories[:3]:
                self.db.update_memory(mem["memory_id"], {"last_recalled_at": datetime.now(timezone.utc).isoformat(), "recall_count": mem.get("recall_count", 0) + 1})
            yield event.plain_result("\n".join(lines))

    @filter.regex(r"^(忘掉|忘记|删除记忆|清除记忆).+")
    async def natural_forget_listener(self, event: AstrMessageEvent):
        message_str = event.message_str.strip()
        content = re.sub(r"^(忘掉|忘记|删除记忆|清除记忆)\s*", "", message_str, count=1)
        if not content: return
        memories = self.db.search_memories(content, limit=3)
        if not memories:
            yield event.plain_result(f"🔍 没有找到关于「{content}」的记忆。")
            return
        for mem in memories:
            self.db.archive_memory(mem["memory_id"], reason="user_forget")
        yield event.plain_result(f"🗑️ 已忘掉关于「{content}」的 {len(memories)} 条记忆。")

    @filter.on_llm_request()
    async def on_llm_request(self, event, *args, **kwargs):
        if not self.auto_listen: return
        try:
            actual_event = getattr(event, 'event', event)
            message_str = getattr(actual_event, 'message_str', '')
            if not message_str: return
            req = kwargs.get('req') or (args[0] if args and hasattr(args[0], 'system_prompt') else None)
            if not req:
                for arg in args:
                    if hasattr(arg, 'system_prompt'): req = arg; break
            if not req: return
            uid = getattr(actual_event, 'unified_msg_origin', None)
            if not uid: return
            conv_mgr = self.context.conversation_manager
            curr_cid = await conv_mgr.get_curr_conversation_id(uid)
            is_new_session = False
            if curr_cid is None:
                is_new_session = True
            else:
                from astrbot.core.conversation_mgr import Conversation
                conversation = await conv_mgr.get_conversation(uid, curr_cid)
                if not conversation or not conversation.history or conversation.history == '[]':
                    is_new_session = True
            if is_new_session:
                strong_memories = self.db.get_strongest_memories(self.max_memories_per_request)
                if strong_memories:
                    memory_text = "\n\n【历史记忆 - 对过去的回忆】\n### 最深刻的记忆：\n"
                    for mem in strong_memories:
                        icon = self._get_emotion_icon(mem.get("emotion", "neutral"))
                        strength = mem.get("strength", 0)
                        label = self.decay_engine.get_strength_label(strength)
                        memory_text += f"{icon} [{strength:.0f}] {mem['content']} ({label})\n"
                    req.system_prompt += memory_text
            hit_keyword = any(kw in message_str for kw in self.recall_keywords)
            hit_prob = random.random() <= self.auto_recall_probability
            if hit_keyword or hit_prob:
                if self._ensure_vector_db():
                    try:
                        search_results = self.vector_db.search_memories(message_str, top_n=self.max_memories_per_request)
                        recalled = []
                        for result in search_results:
                            if result['relevance'] >= self.auto_recall_threshold:
                                mem = self.db.get_memory(result['memory_id'])
                                if mem and mem.get("strength", 0) >= self.strength_threshold:
                                    mem['relevance'] = result['relevance']
                                    recalled.append(mem)
                        if recalled:
                            for mem in recalled:
                                self.db.update_memory(mem["memory_id"], {"last_recalled_at": datetime.now(timezone.utc).isoformat(), "recall_count": mem.get("recall_count", 0) + 1})
                            recall_text = "\n\n【自动回忆 - 相关记忆片段】\n"
                            for mem in recalled[:self.max_memories_per_request]:
                                icon = self._get_emotion_icon(mem.get("emotion", "neutral"))
                                recall_text += f"{icon} {mem['content']}\n"
                            req.system_prompt += recall_text
                    except Exception as e:
                        logger.warning(f"[MemoryWeight] 自动回忆失败: {e}")
        except Exception as e:
            logger.error(f"[MemoryWeight] 注入记忆上下文失败: {e}", exc_info=True)

    async def terminate(self):
        logger.info("[MemoryWeight] plugin shutting down...")
        self._purge_weak_memories()
        logger.info("[MemoryWeight] data saved")

    # ========== auto memory extraction ==========

    def _is_informational(self, text):
        if not text or len(text) < 5:
            return False
        if text.startswith("/"):
            return False
        info_keywords = [
            "我叫", "我是", "我的名字", "我叫什么",
            "我喜欢", "我讨厌", "我不喜欢", "我最爱", "我最讨厌",
            "我住在", "我家在", "我来自",
            "我的生日", "我生日",
            "我工作", "我在", "我的工作", "我的公司",
            "我的朋友", "我的家人", "我爸", "我妈",
            "我会", "我能", "我擅长",
            "我想要", "我需要", "我计划", "我打算",
            "我记得", "我知道",
            "我的猫", "我的狗", "我的宠物",
            "上次", "之前", "以前",
            "今天", "昨天", "明天",
            "开心", "难过", "生气", "担心", "兴奋",
            "项目", "考试", "面试", "出差", "旅行",
            "方案", "问题", "进展",
        ]
        return any(kw in text for kw in info_keywords)

    def _extract_facts_from_text(self, user_msg, ai_msg=""):
        facts = []
        if user_msg and len(user_msg) > 5:
            patterns = [
                (r"\u6211(?:\u53eb|\u7684\u540d\u5b57\u662f)(.+)", "\u8eab\u4efd"),
                (r"\u6211(?:\u559c\u6b22|\u7231|\u6700\u7231)(.+)", "\u504f\u597d"),
                (r"\u6211(?:\u8ba8\u538c|\u4e0d\u559c\u6b22)(.+)", "\u504f\u597d"),
                (r"\u6211(?:\u4f4f\u5728|\u5bb6\u5728|\u6765\u81ea)(.+)", "\u5730\u70b9"),
                (r"\u6211(?:\u7684)?\u751f\u65e5\u662f?(.+)", "\u65f6\u95f4"),
                (r"\u6211(?:\u4f1a|\u80fd|\u64c5\u957f)(.+)", "\u6280\u80fd"),
                (r"\u6211(?:\u60f3\u8981|\u9700\u8981|\u6253\u7b97|\u8ba1\u5212)(.+)", "\u610f\u56fe"),
                (r"\u6211(?:\u7684)?(?:\u732b|\u72d7|\u5ba0\u7269)(?:\u53eb|\u540d\u5b57\u662f)(.+)", "\u5ba0\u7269"),
            ]
            for pattern, tag in patterns:
                match = re.search(pattern, user_msg)
                if match:
                    fact = match.group(1).strip()
                    if fact and len(fact) > 1:
                        facts.append({
                            "content": f"\u7528\u6237{tag}\uff1a{fact}",
                            "memory_type": "semantic",
                            "source": "auto_extract",
                            "tags": [tag],
                            "context": f"\u4ece\u5bf9\u8bdd\u4e2d\u81ea\u52a8\u63d0\u53d6 - \u7528\u6237\u8bf4\uff1a{user_msg[:100]}"
                        })
        return facts

    def _auto_extract_and_store(self, user_msg, ai_msg=""):
        facts = self._extract_facts_from_text(user_msg, ai_msg)
        stored = 0
        for fact in facts:
            try:
                is_new, mem, change = self._add_or_update_memory(
                    content=fact["content"],
                    source=fact.get("source", "auto_extract"),
                    emotion=fact.get("emotion"),
                    emotional_intensity=fact.get("emotional_intensity"),
                    memory_type=fact.get("memory_type", "semantic"),
                    tags=fact.get("tags", []),
                    context=fact.get("context", "")
                )
                if is_new:
                    stored += 1
            except Exception as e:
                logger.warning(f"[MemoryWeight] 自动提取存储失败: {e}")
        return stored

    @filter.on_llm_response()
    async def on_llm_response(self, event, *args, **kwargs):
        if not self.auto_listen:
            return
        try:
            actual_event = getattr(event, 'event', event)
            user_msg = getattr(actual_event, 'message_str', '')
            if not user_msg:
                return
            ai_msg = ""
            resp = kwargs.get('resp') or (args[0] if args else getattr(event, 'resp', None))
            if resp:
                ai_msg = getattr(resp, 'completion_text', '') or ""
            stored = self._auto_extract_and_store(user_msg, ai_msg)
            if stored > 0:
                logger.info(f"[MemoryWeight] 自动从对话中提取了 {stored} 条记忆")
        except Exception as e:
            logger.error(f"[MemoryWeight] auto extract error: {e}", exc_info=True)
