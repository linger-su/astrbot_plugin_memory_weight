# -*- coding: utf-8 -*-
"""
数据库层 - SQLite 存储记忆、节点、关系
"""
import sqlite3
import json
from contextlib import contextmanager
from typing import List, Optional, Dict, Any
from datetime import datetime


class MemoryDB:
    """记忆数据库"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        with self._get_conn() as conn:
            cursor = conn.cursor()
            
            # 核心记忆表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    memory_id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    memory_type TEXT DEFAULT 'episodic',
                    strength REAL DEFAULT 50.0,
                    initial_strength REAL DEFAULT 50.0,
                    stability REAL DEFAULT 1.0,
                    created_at TEXT NOT NULL,
                    last_recalled_at TEXT NOT NULL,
                    recall_count INTEGER DEFAULT 0,
                    consolidation_level INTEGER DEFAULT 0,
                    emotion TEXT DEFAULT 'neutral',
                    emotional_intensity REAL DEFAULT 0.0,
                    tags TEXT DEFAULT '[]',
                    source TEXT DEFAULT 'unknown',
                    context TEXT DEFAULT '',
                    related_memory_ids TEXT DEFAULT '[]',
                    is_archived INTEGER DEFAULT 0,
                    archived_at TEXT
                )
            """)
            
            # 索引
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_memories_strength ON memories(strength)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(memory_type)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_memories_emotion ON memories(emotion)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(created_at)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_memories_archived ON memories(is_archived)")
            
            # 记忆节点表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS memory_nodes (
                    name TEXT PRIMARY KEY,
                    node_type TEXT,
                    description TEXT,
                    last_updated TEXT,
                    importance REAL DEFAULT 5.0,
                    frequency INTEGER DEFAULT 1
                )
            """)
            
            # 记忆关系表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS memory_relations (
                    source_id TEXT,
                    target_id TEXT,
                    relation_type TEXT,
                    confidence REAL DEFAULT 1.0,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (source_id, target_id, relation_type),
                    FOREIGN KEY(source_id) REFERENCES memories(memory_id) ON DELETE CASCADE,
                    FOREIGN KEY(target_id) REFERENCES memories(memory_id) ON DELETE CASCADE
                )
            """)
            
            # 每日摘要表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS daily_digests (
                    date TEXT PRIMARY KEY,
                    memory_count INTEGER DEFAULT 0,
                    recalled_count INTEGER DEFAULT 0,
                    avg_strength REAL DEFAULT 0.0,
                    top_memories TEXT DEFAULT '[]',
                    daily_reflection TEXT DEFAULT ''
                )
            """)
            
            # 回收站（已归档记忆）
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS recycle_bin (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    memory_id TEXT,
                    content TEXT,
                    archived_at TEXT,
                    initial_strength REAL,
                    recall_count INTEGER,
                    source TEXT,
                    reason TEXT DEFAULT 'strength_zero'
                )
            """)
            
            conn.commit()
    
    @contextmanager
    def _get_conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    # ========== 记忆 CRUD ==========
    
    def insert_memory(self, memory: dict) -> bool:
        """插入新记忆"""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO memories 
                (memory_id, content, memory_type, strength, initial_strength, stability,
                 created_at, last_recalled_at, recall_count, consolidation_level,
                 emotion, emotional_intensity, tags, source, context, related_memory_ids)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                memory["memory_id"],
                memory["content"],
                memory.get("memory_type", "episodic"),
                memory.get("strength", 50.0),
                memory.get("initial_strength", 50.0),
                memory.get("stability", 1.0),
                memory["created_at"],
                memory.get("last_recalled_at", memory["created_at"]),
                memory.get("recall_count", 0),
                memory.get("consolidation_level", 0),
                memory.get("emotion", "neutral"),
                memory.get("emotional_intensity", 0.0),
                json.dumps(memory.get("tags", []), ensure_ascii=False),
                memory.get("source", "unknown"),
                memory.get("context", ""),
                json.dumps(memory.get("related_memory_ids", []), ensure_ascii=False)
            ))
            conn.commit()
            return True
    
    def get_memory(self, memory_id: str) -> Optional[dict]:
        """获取单条记忆"""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM memories WHERE memory_id = ?", (memory_id,))
            row = cursor.fetchone()
            if row:
                return self._row_to_dict(row)
            return None
    
    def get_all_memories(self, include_archived: bool = False) -> List[dict]:
        """获取所有记忆"""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            if include_archived:
                cursor.execute("SELECT * FROM memories ORDER BY created_at DESC")
            else:
                cursor.execute("SELECT * FROM memories WHERE is_archived = 0 ORDER BY created_at DESC")
            return [self._row_to_dict(row) for row in cursor.fetchall()]
    
    def update_memory(self, memory_id: str, updates: dict) -> bool:
        """更新记忆字段"""
        if not updates:
            return False
        with self._get_conn() as conn:
            cursor = conn.cursor()
            set_clauses = []
            values = []
            for key, value in updates.items():
                if key in ("tags", "related_memory_ids") and isinstance(value, list):
                    value = json.dumps(value, ensure_ascii=False)
                set_clauses.append(f"{key} = ?")
                values.append(value)
            values.append(memory_id)
            cursor.execute(f"UPDATE memories SET {', '.join(set_clauses)} WHERE memory_id = ?", values)
            conn.commit()
            return cursor.rowcount > 0
    
    def archive_memory(self, memory_id: str, reason: str = "strength_zero") -> bool:
        """归档记忆（移到回收站）"""
        mem = self.get_memory(memory_id)
        if not mem:
            return False
        
        with self._get_conn() as conn:
            cursor = conn.cursor()
            now = datetime.now().isoformat()
            
            # 插入回收站
            cursor.execute("""
                INSERT INTO recycle_bin (memory_id, content, archived_at, initial_strength, recall_count, source, reason)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (memory_id, mem["content"], now, mem.get("initial_strength", 50), 
                  mem.get("recall_count", 0), mem.get("source", ""), reason))
            
            # 标记归档
            cursor.execute("UPDATE memories SET is_archived = 1, archived_at = ? WHERE memory_id = ?", (now, memory_id))
            conn.commit()
            return True
    
    def search_memories(self, query: str, limit: int = 10, min_strength: float = 0.0) -> List[dict]:
        """模糊搜索记忆"""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM memories 
                WHERE is_archived = 0 
                AND strength >= ?
                AND (content LIKE ? OR tags LIKE ? OR context LIKE ?)
                ORDER BY strength DESC, created_at DESC
                LIMIT ?
            """, (min_strength, f"%{query}%", f"%{query}%", f"%{query}%", limit))
            return [self._row_to_dict(row) for row in cursor.fetchall()]
    
    def get_memories_by_emotion(self, emotion: str, limit: int = 10) -> List[dict]:
        """按情感类型获取记忆"""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM memories 
                WHERE is_archived = 0 AND emotion = ?
                ORDER BY emotional_intensity DESC, strength DESC
                LIMIT ?
            """, (emotion, limit))
            return [self._row_to_dict(row) for row in cursor.fetchall()]
    
    def get_strongest_memories(self, limit: int = 10) -> List[dict]:
        """获取最强记忆"""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM memories 
                WHERE is_archived = 0
                ORDER BY strength DESC
                LIMIT ?
            """, (limit,))
            return [self._row_to_dict(row) for row in cursor.fetchall()]
    
    def get_weakest_memories(self, limit: int = 10) -> List[dict]:
        """获取最弱记忆（即将遗忘的）"""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM memories 
                WHERE is_archived = 0
                ORDER BY strength ASC
                LIMIT ?
            """, (limit,))
            return [self._row_to_dict(row) for row in cursor.fetchall()]
    
    def count_memories(self) -> dict:
        """统计记忆数量"""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) as total FROM memories WHERE is_archived = 0")
            total = cursor.fetchone()["total"]
            cursor.execute("SELECT COUNT(*) as archived FROM memories WHERE is_archived = 1")
            archived = cursor.fetchone()["archived"]
            cursor.execute("SELECT AVG(strength) as avg_strength FROM memories WHERE is_archived = 0")
            avg_row = cursor.fetchone()
            avg_strength = avg_row["avg_strength"] if avg_row and avg_row["avg_strength"] else 0.0
            return {
                "total": total,
                "archived": archived,
                "active": total,
                "avg_strength": round(avg_strength, 2)
            }
    
    # ========== 节点 CRUD ==========
    
    def update_node(self, node: dict) -> bool:
        """更新记忆节点"""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO memory_nodes (name, node_type, description, last_updated, importance, frequency)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(name) DO UPDATE SET
                    node_type = excluded.node_type,
                    description = excluded.description,
                    last_updated = excluded.last_updated,
                    importance = excluded.importance,
                    frequency = memory_nodes.frequency + 1
            """, (
                node["name"], node.get("node_type", ""), node.get("description", ""),
                node.get("last_updated", datetime.now().isoformat()),
                node.get("importance", 5.0), node.get("frequency", 1)
            ))
            conn.commit()
            return True
    
    def search_nodes(self, query: str, limit: int = 5) -> List[dict]:
        """搜索记忆节点"""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM memory_nodes 
                WHERE name LIKE ? OR description LIKE ?
                ORDER BY importance DESC, frequency DESC
                LIMIT ?
            """, (f"%{query}%", f"%{query}%", limit))
            return [dict(row) for row in cursor.fetchall()]
    
    def get_node(self, name: str) -> Optional[dict]:
        """获取单个节点"""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM memory_nodes WHERE name = ?", (name,))
            row = cursor.fetchone()
            return dict(row) if row else None
    
    def delete_node(self, name: str) -> bool:
        """删除节点"""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM memory_nodes WHERE name = ?", (name,))
            conn.commit()
            return cursor.rowcount > 0
    
    # ========== 关系 CRUD ==========
    
    def add_relation(self, source_id: str, target_id: str, relation_type: str, confidence: float = 1.0) -> bool:
        """添加记忆关系"""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO memory_relations (source_id, target_id, relation_type, confidence)
                VALUES (?, ?, ?, ?)
            """, (source_id, target_id, relation_type, confidence))
            conn.commit()
            return True
    
    def get_related_memories(self, memory_id: str) -> List[dict]:
        """获取关联记忆"""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT m.*, mr.relation_type, mr.confidence 
                FROM memories m
                JOIN memory_relations mr ON (m.memory_id = mr.target_id OR m.memory_id = mr.source_id)
                WHERE (mr.source_id = ? OR mr.target_id = ?)
                AND m.is_archived = 0
                ORDER BY mr.confidence DESC
            """, (memory_id, memory_id))
            return [self._row_to_dict(row) for row in cursor.fetchall()]
    
    def has_relation(self, id1: str, id2: str) -> bool:
        """检查两个记忆是否有关系"""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT 1 FROM memory_relations 
                WHERE (source_id = ? AND target_id = ?) OR (source_id = ? AND target_id = ?)
            """, (id1, id2, id2, id1))
            return cursor.fetchone() is not None
    
    # ========== 回收站 ==========
    
    def get_recycle_bin(self, limit: int = 20) -> List[dict]:
        """获取回收站内容"""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM recycle_bin ORDER BY archived_at DESC LIMIT ?", (limit,))
            return [dict(row) for row in cursor.fetchall()]
    
    def restore_from_bin(self, memory_id: str) -> bool:
        """从回收站恢复"""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("UPDATE memories SET is_archived = 0, archived_at = NULL WHERE memory_id = ?", (memory_id,))
            cursor.execute("DELETE FROM recycle_bin WHERE memory_id = ?", (memory_id,))
            conn.commit()
            return cursor.rowcount > 0
    
    def clear_recycle_bin(self) -> int:
        """清空回收站"""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) as count FROM recycle_bin")
            count = cursor.fetchone()["count"]
            cursor.execute("DELETE FROM recycle_bin")
            conn.commit()
            return count
    
    # ========== 每日摘要 ==========
    
    def save_daily_digest(self, digest: dict) -> bool:
        """保存每日摘要"""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO daily_digests 
                (date, memory_count, recalled_count, avg_strength, top_memories, daily_reflection)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                digest["date"], digest.get("memory_count", 0),
                digest.get("recalled_count", 0), digest.get("avg_strength", 0.0),
                json.dumps(digest.get("top_memories", []), ensure_ascii=False),
                digest.get("daily_reflection", "")
            ))
            conn.commit()
            return True
    
    def get_daily_digest(self, date: str) -> Optional[dict]:
        """获取每日摘要"""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM daily_digests WHERE date = ?", (date,))
            row = cursor.fetchone()
            if row:
                d = dict(row)
                d["top_memories"] = json.loads(d.get("top_memories", "[]"))
                return d
            return None
    
    def get_recent_digests(self, days: int = 7) -> List[dict]:
        """获取最近N天的摘要"""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM daily_digests 
                ORDER BY date DESC LIMIT ?
            """, (days,))
            results = []
            for row in cursor.fetchall():
                d = dict(row)
                d["top_memories"] = json.loads(d.get("top_memories", "[]"))
                results.append(d)
            return results
    
    # ========== 工具方法 ==========
    
    def _row_to_dict(self, row) -> dict:
        """将Row对象转换为字典"""
        d = dict(row)
        # 解析JSON字段
        for field in ("tags", "related_memory_ids"):
            if field in d and isinstance(d[field], str):
                try:
                    d[field] = json.loads(d[field])
                except (json.JSONDecodeError, TypeError):
                    d[field] = []
        return d
