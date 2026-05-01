# -*- coding: utf-8 -*-
"""
记忆衰减引擎 - 模拟人类遗忘曲线
核心算法：Ebbinghaus遗忘曲线 + 间隔重复强化
"""
import math
from datetime import datetime, timezone, timedelta
from typing import Optional


class MemoryDecayEngine:
    """
    记忆衰减引擎
    
    模拟人类记忆的自然衰减过程：
    1. Ebbinghaus遗忘曲线：R = e^(-t/S)
       - R = 记忆保留率
       - t = 距离上次回忆的时间
       - S = 记忆稳定性（越高越不容易忘）
    
    2. 间隔重复效应：每次回忆都会增加稳定性
       - stability *= (1 + boost_factor)
    
    3. 情感增强：高情感强度的记忆更稳定
       - stability *= (1 + emotional_bonus)
    
    4. 巩固效应：巩固等级越高，衰减越慢
    """
    
    # 默认参数
    DEFAULT_BASE_STABILITY = 1.0        # 基础稳定性
    DEFAULT_MIN_STRENGTH = 0.0          # 最低强度（低于此值视为遗忘）
    DEFAULT_RECALL_BOOST = 0.15         # 每次回忆的稳定性提升系数
    DEFAULT_EMOTIONAL_BONUS_MAX = 0.5   # 情感增强最大系数
    DEFAULT_CONSOLIDATION_BONUS = 0.1   # 每级巩固的稳定性加成
    DEFAULT_MIN_RECALL_INTERVAL_HOURS = 1  # 最小回忆间隔（小时）
    
    def __init__(self, config: dict = None):
        config = config or {}
        self.recall_boost = config.get("recall_boost", self.DEFAULT_RECALL_BOOST)
        self.emotional_bonus_max = config.get("emotional_bonus_max", self.DEFAULT_EMOTIONAL_BONUS_MAX)
        self.consolidation_bonus = config.get("consolidation_bonus", self.DEFAULT_CONSOLIDATION_BONUS)
        self.min_recall_interval_hours = config.get("min_recall_interval_hours", self.DEFAULT_MIN_RECALL_INTERVAL_HOURS)
        self.min_strength = config.get("min_strength", self.DEFAULT_MIN_STRENGTH)
    
    def calculate_strength(
        self,
        initial_strength: float,
        stability: float,
        last_recalled_at: str,
        consolidation_level: int = 0,
        emotional_intensity: float = 0.0,
        recall_count: int = 0
    ) -> float:
        """
        计算当前记忆强度
        
        使用修正的Ebbinghaus遗忘曲线：
        strength = initial_strength * e^(-t / (S * consolidation_factor * emotion_factor))
        
        Args:
            initial_strength: 初始强度 (0-100)
            stability: 记忆稳定性
            last_recalled_at: 上次回忆时间 (ISO格式)
            consolidation_level: 巩固等级 (0-5)
            emotional_intensity: 情感强度 (0-10)
            recall_count: 回忆次数
        
        Returns:
            当前记忆强度 (0-100)
        """
        try:
            last_recalled = datetime.fromisoformat(last_recalled_at)
            now = datetime.now(timezone.utc).replace(tzinfo=None)
        except (ValueError, TypeError):
            return initial_strength
        
        # 计算时间差（小时）
        elapsed_hours = max(0, (now - last_recalled).total_seconds() / 3600.0)
        
        if elapsed_hours <= 0:
            return initial_strength
        
        # 计算有效稳定性
        effective_stability = self._calculate_effective_stability(
            base_stability=stability,
            consolidation_level=consolidation_level,
            emotional_intensity=emotional_intensity,
            recall_count=recall_count
        )
        
        # Ebbinghaus遗忘曲线: R = e^(-t/S)
        # strength = initial * R
        retention_rate = math.exp(-elapsed_hours / (effective_stability * 24.0))  # 按天计算
        current_strength = initial_strength * retention_rate
        
        return max(self.min_strength, min(100.0, current_strength))
    
    def _calculate_effective_stability(
        self,
        base_stability: float,
        consolidation_level: int,
        emotional_intensity: float,
        recall_count: int
    ) -> float:
        """
        计算有效稳定性（综合多个因素）
        
        有效稳定性 = 基础稳定性 
                    * (1 + 巩固加成) 
                    * (1 + 情感加成)
                    * (1 + 回忆次数加成)
        """
        # 1. 巩固加成：每级巩固增加稳定性
        consolidation_factor = 1.0 + (consolidation_level * self.consolidation_bonus)
        
        # 2. 情感加成：情感越强，记忆越稳定
        # 使用对数函数，避免情感过强导致的不合理结果
        emotional_factor = 1.0 + (
            min(emotional_intensity / 10.0, 1.0) * self.emotional_bonus_max
        )
        
        # 3. 回忆次数加成：回忆越多，越不容易忘
        # 使用递减加成，避免无限增长
        recall_factor = 1.0 + min(recall_count * 0.05, 0.5)
        
        effective = base_stability * consolidation_factor * emotional_factor * recall_factor
        return max(0.1, effective)
    
    def apply_recall_boost(
        self,
        current_stability: float,
        current_strength: float,
        recall_count: int
    ) -> tuple[float, float]:
        """
        应用回忆强化效果
        
        每次回忆都会：
        1. 提高记忆稳定性（使遗忘曲线变平缓）
        2. 恢复部分记忆强度
        
        Returns:
            (new_stability, new_strength)
        """
        # 稳定性提升（递减效果）
        boost = self.recall_boost * (1.0 / (1.0 + recall_count * 0.1))
        new_stability = current_stability * (1.0 + boost)
        
        # 强度恢复：恢复到初始强度的一定比例
        # 回忆次数越多，恢复比例越高（但有上限）
        recovery_rate = min(0.3 + recall_count * 0.02, 0.8)
        new_strength = min(100.0, current_strength + (100.0 - current_strength) * recovery_rate)
        
        return new_stability, new_strength
    
    def should_consolidate(self, memory: dict) -> bool:
        """
        判断记忆是否应该进入下一级巩固
        
        巩固条件：
        1. 回忆次数 >= 阈值
        2. 记忆强度 >= 阈值
        3. 存在时间 >= 最短期限
        """
        recall_count = memory.get("recall_count", 0)
        strength = memory.get("strength", 0)
        consolidation_level = memory.get("consolidation_level", 0)
        
        # 每级巩固需要的最低回忆次数（递增）
        min_recalls = [2, 5, 10, 20, 50]
        # 每级巩固需要的最低强度
        min_strengths = [60, 65, 70, 75, 80]
        
        if consolidation_level >= 5:
            return False
        
        threshold_recalls = min_recalls[consolidation_level]
        threshold_strength = min_strengths[consolidation_level]
        
        return recall_count >= threshold_recalls and strength >= threshold_strength
    
    def get_strength_label(self, strength: float) -> str:
        """根据强度返回人类可读的标签"""
        if strength >= 80:
            return "深刻记忆"
        elif strength >= 60:
            return "清晰记忆"
        elif strength >= 40:
            return "一般记忆"
        elif strength >= 20:
            return "模糊记忆"
        elif strength >= 5:
            return "即将遗忘"
        else:
            return "已遗忘"
    
    def get_forgetting_time(self, initial_strength: float, stability: float) -> float:
        """
        计算预计遗忘时间（天数）
        
        返回从创建到强度降到5以下的预计天数
        """
        if stability <= 0:
            return 0
        
        # R = e^(-t/S) = 5/initial_strength
        # t = -S * ln(5/initial_strength)
        target_retention = 5.0 / max(initial_strength, 1.0)
        if target_retention >= 1.0:
            return float('inf')
        
        days = -stability * math.log(target_retention)
        return max(0, days)
    
    def calculate_batch_strengths(self, memories: list) -> list:
        """
        批量计算记忆强度
        
        Args:
            memories: 记忆字典列表
        
        Returns:
            更新后的记忆列表（strength字段已更新）
        """
        for mem in memories:
            mem["strength"] = self.calculate_strength(
                initial_strength=mem.get("initial_strength", 50.0),
                stability=mem.get("stability", 1.0),
                last_recalled_at=mem.get("last_recalled_at", mem.get("created_at", "")),
                consolidation_level=mem.get("consolidation_level", 0),
                emotional_intensity=mem.get("emotional_intensity", 0.0),
                recall_count=mem.get("recall_count", 0)
            )
        return memories
