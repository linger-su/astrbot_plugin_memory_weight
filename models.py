# -*- coding: utf-8 -*-
"""
数据模型 - 记忆权重插件
定义记忆条目、记忆节点、记忆关系等核心数据结构
"""
from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum


class MemoryType(str, Enum):
    """记忆类型"""
    EPISODIC = "episodic"       # 情景记忆（具体事件）
    SEMANTIC = "semantic"       # 语义记忆（知识/概念）
    PROCEDURAL = "procedural"   # 程序记忆（习惯/技能）
    EMOTIONAL = "emotional"     # 情感记忆（情绪体验）


class EmotionTag(str, Enum):
    """情感标签"""
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    FEARFUL = "fearful"
    SURPRISED = "surprised"
    NEUTRAL = "neutral"
    EXCITED = "excited"
    GRATEFUL = "grateful"
    NOSTALGIC = "nostalgic"
    PROUD = "proud"
    ANXIOUS = "anxious"
    PEACEFUL = "peaceful"


class Memory(BaseModel):
    """单条记忆"""
    memory_id: str = Field(description="唯一ID，格式 mem_YYYYMMDD_NNN")
    content: str = Field(description="记忆内容（自然语言描述）")
    memory_type: MemoryType = Field(default=MemoryType.EPISODIC, description="记忆类型")
    
    # 权重系统
    strength: float = Field(default=50.0, ge=0.0, le=100.0, description="当前记忆强度 0-100")
    initial_strength: float = Field(default=50.0, description="初始强度")
    stability: float = Field(default=1.0, ge=0.1, le=100.0, description="稳定性（衰减速度倒数，越高越不容易忘）")
    
    # 时间信息
    created_at: str = Field(description="创建时间 ISO格式")
    last_recalled_at: str = Field(description="最后回忆时间 ISO格式")
    recall_count: int = Field(default=0, description="被回忆次数")
    consolidation_level: int = Field(default=0, ge=0, le=5, description="巩固等级 0=短期 5=深度长期")
    
    # 情感与上下文
    emotion: EmotionTag = Field(default=EmotionTag.NEUTRAL, description="情感标签")
    emotional_intensity: float = Field(default=0.0, ge=0.0, le=10.0, description="情感强度 0-10")
    tags: List[str] = Field(default_factory=list, description="标签列表")
    source: str = Field(default="unknown", description="来源（user_input/ai_observation/auto_extract）")
    context: str = Field(default="", description="上下文信息")
    
    # 关联
    related_memory_ids: List[str] = Field(default_factory=list, description="关联的记忆ID列表")
    
    class Config:
        json_schema_extra = {
            "example": {
                "memory_id": "mem_20260501_001",
                "content": "用户说他喜欢喝冰美式",
                "memory_type": "semantic",
                "strength": 75.0,
                "emotion": "neutral",
                "tags": ["偏好", "饮品"]
            }
        }


class MemoryNode(BaseModel):
    """记忆节点（实体/概念）"""
    name: str = Field(description="节点名称（实体词，如：小明、北京、Python）")
    node_type: str = Field(description="节点类型（人物、地点、技术、情感等）")
    description: str = Field(description="对该节点的综合描述")
    last_updated: str = Field(description="最后更新时间")
    
    # 节点权重
    importance: float = Field(default=5.0, ge=0.0, le=10.0, description="重要性 0-10")
    frequency: int = Field(default=1, description="被提及次数")


class MemoryRelation(BaseModel):
    """记忆之间的关系"""
    source_id: str = Field(description="源记忆ID")
    target_id: str = Field(description="目标记忆ID")
    relation_type: str = Field(description="关系类型：caused_by/related_to/context_for/contradicts/extends")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="置信度")


class DailyDigest(BaseModel):
    """每日记忆摘要"""
    date: str = Field(description="日期 YYYY-MM-DD")
    memory_count: int = Field(description="当日新增记忆数")
    recalled_count: int = Field(description="当日被回忆的记忆数")
    avg_strength: float = Field(description="当日记忆平均强度")
    top_memories: List[str] = Field(default_factory=list, description="最重要的记忆内容摘要")
    daily_reflection: str = Field(default="", description="每日反思/感悟")
