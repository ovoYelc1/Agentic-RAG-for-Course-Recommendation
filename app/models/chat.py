# 数据模型定义
from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum


class Intent(str, Enum):
    """意图枚举"""
    FAQ = "faq"
    COURSE_RECOMMEND = "course_recommend"
    ORDER_QUERY = "order_query"
    ACCOUNT_HELP = "account_help"
    GENERIC = "generic"


class ChatRequest(BaseModel):
    """聊天请求"""
    message: str = Field(..., min_length=1, max_length=500, description="用户消息")
    user_id: Optional[str] = Field(None, description="用户ID（用于个性化推荐）")
    session_id: Optional[str] = Field(None, description="会话ID（用于上下文记忆）")


class ChatResponse(BaseModel):
    """聊天响应"""
    intent: Intent = Field(description="识别的意图")
    answer: str = Field(description="AI 回答")
    courses: Optional[List[dict]] = Field(default=None, description="推荐课程列表（仅推荐意图时有值）")
    confidence: float = Field(description="意图识别置信度 0-1")


class HistoryItem(BaseModel):
    """单条聊天历史（前端恢复会话用，口径同 memory.get_history include_full）"""
    message_id: str = Field(description="消息ID（幂等）")
    role: str = Field(description="user / assistant")
    content: str = Field(description="消息文本")
    created_at: Optional[int] = Field(default=None, description="创建时间戳")
    expires_at: Optional[int] = Field(default=None, description="过期时间戳")


class HistoryResponse(BaseModel):
    """会话历史响应"""
    conversation_id: str = Field(description="当前会话ID（与 chat 请求同源）")
    messages: List[HistoryItem] = Field(default_factory=list, description="未过期消息（按创建时间升序）")
