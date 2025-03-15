"""
models/conversation.py
Defines conversation data models and handles message serialization.
"""
from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict, Any, Union
from uuid import uuid4, UUID

from pydantic import BaseModel, Field, field_validator, ConfigDict


class MessageRole(str, Enum):
    """Roles for conversation messages."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class MessageStatus(str, Enum):
    """Status of message processing."""
    PENDING = "pending"
    GENERATING = "generating"
    COMPLETE = "complete"
    ERROR = "error"


class Message(BaseModel):
    """Model representing a single message in a conversation."""
    
    model_config = ConfigDict(populate_by_name=True)
    
    id: UUID = Field(default_factory=uuid4)
    role: MessageRole
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)
    status: MessageStatus = Field(default=MessageStatus.COMPLETE)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @field_validator('content')
    @classmethod
    def content_not_empty(cls, value: str) -> str:
        """Validate that message content is not empty."""
        if not value.strip():
            raise ValueError("Message content cannot be empty")
        return value


class Conversation(BaseModel):
    """Model representing a conversation with multiple messages."""
    
    model_config = ConfigDict(populate_by_name=True)
    
    id: UUID = Field(default_factory=uuid4)
    title: str = "New Conversation"
    messages: List[Message] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def add_message(self, role: MessageRole, content: str, status: MessageStatus = MessageStatus.COMPLETE) -> Message:
        """
        Add a new message to the conversation.
        
        Args:
            role: Role of the message sender
            content: Message content
            status: Message processing status
            
        Returns:
            Message: The newly created message
        """
        message = Message(
            role=role,
            content=content,
            status=status
        )
        self.messages.append(message)
        self.updated_at = datetime.now()
        return message
    
    def add_user_message(self, content: str) -> Message:
        """
        Add a user message to the conversation.
        
        Args:
            content: Message content
            
        Returns:
            Message: The newly created message
        """
        return self.add_message(MessageRole.USER, content)
    
    def add_assistant_message(self, content: str, status: MessageStatus = MessageStatus.COMPLETE) -> Message:
        """
        Add an assistant message to the conversation.
        
        Args:
            content: Message content
            status: Message processing status
            
        Returns:
            Message: The newly created message
        """
        return self.add_message(MessageRole.ASSISTANT, content, status)
    
    def add_system_message(self, content: str) -> Message:
        """
        Add a system message to the conversation.
        
        Args:
            content: Message content
            
        Returns:
            Message: The newly created message
        """
        return self.add_message(MessageRole.SYSTEM, content)
    
    def get_message(self, message_id: Union[str, UUID]) -> Optional[Message]:
        """
        Get a message by its ID.
        
        Args:
            message_id: ID of the message to retrieve
            
        Returns:
            Optional[Message]: The message if found, None otherwise
        """
        if isinstance(message_id, str):
            try:
                message_id = UUID(message_id)
            except ValueError:
                return None
                
        for message in self.messages:
            if message.id == message_id:
                return message
        return None
    
    def update_message(self, message_id: Union[str, UUID], content: str = None, status: MessageStatus = None) -> bool:
        """
        Update a message in the conversation.
        
        Args:
            message_id: ID of the message to update
            content: New message content (optional)
            status: New message status (optional)
            
        Returns:
            bool: True if the message was updated, False otherwise
        """
        message = self.get_message(message_id)
        if not message:
            return False
            
        if content is not None:
            message.content = content
            
        if status is not None:
            message.status = status
            
        self.updated_at = datetime.now()
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the conversation to a dictionary.
        
        Returns:
            Dict[str, Any]: Dictionary representation of the conversation
        """
        return self.model_dump()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Conversation":
        """
        Create a conversation from a dictionary.
        
        Args:
            data: Dictionary representation of the conversation
            
        Returns:
            Conversation: The created conversation
        """
        return cls.model_validate(data)
    

class ConversationState:
    """
    Singleton class to manage the current conversation state.
    """
    _instance = None
    _current_conversation: Optional[Conversation] = None
    _history: List[Conversation] = []
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConversationState, cls).__new__(cls)
            cls._current_conversation = None
            cls._history = []
        return cls._instance
    
    @property
    def current_conversation(self) -> Optional[Conversation]:
        """Get the current conversation."""
        if self._current_conversation is None:
            self.new_conversation()
        return self._current_conversation
    
    @property
    def history(self) -> List[Conversation]:
        """Get the conversation history."""
        return self._history
    
    def new_conversation(self, title: str = "New Conversation") -> Conversation:
        """
        Create a new conversation and set it as current.
        
        Args:
            title: Title for the new conversation
            
        Returns:
            Conversation: The newly created conversation
        """
        # If there's a current conversation, add it to history if it has messages
        if self._current_conversation and self._current_conversation.messages:
            if self._current_conversation not in self._history:
                self._history.append(self._current_conversation)
        
        # Create new conversation
        self._current_conversation = Conversation(title=title)
        return self._current_conversation
    
    def load_conversation(self, conversation_id: Union[str, UUID]) -> Optional[Conversation]:
        """
        Load a conversation from history by its ID.
        
        Args:
            conversation_id: ID of the conversation to load
            
        Returns:
            Optional[Conversation]: The loaded conversation if found, None otherwise
        """
        if isinstance(conversation_id, str):
            try:
                conversation_id = UUID(conversation_id)
            except ValueError:
                return None
        
        for conversation in self._history:
            if conversation.id == conversation_id:
                self._current_conversation = conversation
                return conversation
        return None
    
    def add_user_message(self, content: str) -> Message:
        """
        Add a user message to the current conversation.
        
        Args:
            content: Message content
            
        Returns:
            Message: The newly created message
        """
        return self.current_conversation.add_user_message(content)
    
    def add_assistant_message(self, content: str, status: MessageStatus = MessageStatus.COMPLETE) -> Message:
        """
        Add an assistant message to the current conversation.
        
        Args:
            content: Message content
            status: Message processing status
            
        Returns:
            Message: The newly created message
        """
        return self.current_conversation.add_assistant_message(content, status)