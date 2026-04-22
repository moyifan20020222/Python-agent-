"""
Kafka生产者模块 - 用于异步处理LangGraph总结任务
"""

from kafka import KafkaProducer
import json
import os
from typing import Dict, Any, Optional
import uuid
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class KafkaTaskProducer:
    """Kafka任务生产者"""
    
    def __init__(self):
        # Kafka配置（可从环境变量读取）
        self.bootstrap_servers = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
        self.topic = os.getenv("KAFKA_TOPIC", "rehab_summary_tasks")
        
        # 创建生产者
        try:
            self.producer = KafkaProducer(
                bootstrap_servers=self.bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                key_serializer=str.encode,
                acks='all',  # 确保消息可靠投递
                retries=3,
                retry_backoff_ms=1000
            )
            logger.info(f"✅ Kafka生产者初始化成功: {self.bootstrap_servers}")
        except Exception as e:
            logger.error(f"❌ Kafka生产者初始化失败: {e}")
            self.producer = None
    
    def send_finalize_task(self, 
                          session_id: str,
                          case_id: str,
                          patient_id: str,
                          context_data: Dict[str, Any],
                          priority: str = "normal") -> str:
        """
        发送总结任务到Kafka
        返回 execution_id 用于状态跟踪
        """
        if not self.producer:
            raise RuntimeError("Kafka生产者未初始化")
        
        # 生成唯一执行ID
        execution_id = f"exec_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        
        # 构建消息
        message = {
            "execution_id": execution_id,
            "session_id": session_id,
            "case_id": case_id,
            "patient_id": patient_id,
            "action": "finalize",
            "context": context_data,
            "created_at": datetime.now().isoformat(),
            "priority": priority,
            "status": "queued"
        }
        
        try:
            # 发送消息到Kafka
            future = self.producer.send(
                self.topic,
                key=session_id.encode('utf-8'),  # 按session_id分区
                value=message
            )
            
            # 等待发送完成（开发环境，生产环境可异步）
            record_metadata = future.get(timeout=10)
            logger.info(f"✅ 总结任务已发送到Kafka: {execution_id} (partition: {record_metadata.partition})")
            
            return execution_id
            
        except Exception as e:
            logger.error(f"❌ 发送Kafka消息失败: {e}")
            raise

# 全局实例（在main.py中初始化）
kafka_producer = None