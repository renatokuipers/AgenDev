"""
components/conversation.py
Implements conversation panel, message history display, and conversation controls.
"""
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFrame, QLabel, 
    QPushButton, QTextEdit, QLineEdit, QScrollArea, 
    QSizePolicy, QSpacerItem, QMenu
)
from PyQt6.QtCore import Qt, QSize, pyqtSignal, pyqtSlot, QTimer
from PyQt6.QtGui import QIcon, QFont, QColor, QTextCursor, QKeySequence, QShortcut, QAction
import html
import time

# Import models
from models.conversation import (
    Conversation, Message, MessageRole, MessageStatus, ConversationState
)


class MessageBubble(QFrame):
    """Widget representing a single message bubble in the conversation."""
    
    def __init__(self, message: Message, parent=None):
        """Initialize message bubble with message data.
        
        Args:
            message: Message object to display
            parent: Parent widget
        """
        super().__init__(parent)
        self.message = message
        self.setObjectName("messageBubble")
        
        # Set up frame appearance
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setFrameShadow(QFrame.Shadow.Raised)
        
        # Apply appropriate styling based on message role
        if message.role == MessageRole.USER:
            self.setObjectName("userMessage")
            self.setStyleSheet("""
                QFrame#userMessage {
                    background-color: #383838;
                    border-radius: 8px;
                    margin: 4px 48px 4px 4px;
                }
            """)
        else:
            self.setObjectName("assistantMessage")
            self.setStyleSheet("""
                QFrame#assistantMessage {
                    background-color: #2A2A2A;
                    border-radius: 8px;
                    margin: 4px 4px 4px 48px;
                }
            """)
        
        self.setup_ui()
    
    def setup_ui(self):
        """Set up the message bubble UI."""
        # Main layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(4)
        
        # Header with role and timestamp
        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 4)
        
        # Role label
        role_name = self.message.role.value.capitalize()
        role_label = QLabel(role_name)
        role_label.setObjectName("roleLabel")
        
        # Format and display timestamp
        timestamp_str = self.message.timestamp.strftime("%H:%M")
        timestamp_label = QLabel(timestamp_str)
        timestamp_label.setObjectName("timestampLabel")
        timestamp_label.setStyleSheet("color: #999999; font-size: 9pt;")
        
        header_layout.addWidget(role_label)
        header_layout.addStretch()
        header_layout.addWidget(timestamp_label)
        
        # Message content
        content_label = QTextEdit()
        content_label.setObjectName("contentLabel")
        content_label.setReadOnly(True)
        content_label.setHtml(self._format_message_content(self.message.content))
        content_label.setFrameStyle(QFrame.Shape.NoFrame)
        content_label.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        content_label.setStyleSheet("""
            QTextEdit {
                background-color: transparent;
                border: none;
            }
        """)
        
        # Set a reasonable minimum and maximum height for the content
        content_label.setMinimumHeight(20)
        content_label.setMaximumHeight(300)
        content_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, 
            QSizePolicy.Policy.Preferred
        )
        
        # Adjust content text size to document size
        doc_height = content_label.document().size().height()
        content_label.setFixedHeight(min(max(doc_height + 10, 20), 300))
        
        # Add widgets to main layout
        layout.addLayout(header_layout)
        layout.addWidget(content_label)
        
        # Add status indicator for assistant messages
        if self.message.role == MessageRole.ASSISTANT:
            status_layout = QHBoxLayout()
            
            if self.message.status == MessageStatus.GENERATING:
                status_label = QLabel("Generating...")
                status_label.setObjectName("generatingLabel")
                status_label.setStyleSheet("color: #76D7C4; font-size: 9pt; font-style: italic;")
                status_layout.addWidget(status_label)
                
            elif self.message.status == MessageStatus.ERROR:
                error_label = QLabel("Error occurred")
                error_label.setObjectName("errorLabel")
                error_label.setStyleSheet("color: #E74C3C; font-size: 9pt;")
                status_layout.addWidget(error_label)
                
            layout.addLayout(status_layout)
    
    def _format_message_content(self, content: str) -> str:
        """Format the message content with HTML.
        
        Args:
            content: Raw message content
            
        Returns:
            str: HTML formatted content
        """
        # Escape HTML special characters
        content = html.escape(content)
        
        # Convert newlines to <br> tags
        content = content.replace("\n", "<br>")
        
        # Format code blocks
        # This is a simple implementation - a more robust one would use a proper markdown parser
        if "```" in content:
            parts = content.split("```")
            formatted_parts = []
            
            for i, part in enumerate(parts):
                if i % 2 == 0:
                    # Regular text
                    formatted_parts.append(part)
                else:
                    # Code block
                    code_style = "background-color: #1E1E1E; padding: 8px; border-radius: 4px; font-family: monospace; white-space: pre;"
                    formatted_code = f'<div style="{code_style}">{part}</div>'
                    formatted_parts.append(formatted_code)
            
            content = "".join(formatted_parts)
        
        return content


class MessageInput(QLineEdit):
    """Input field for entering conversation messages."""
    
    message_submitted = pyqtSignal(str)
    
    def __init__(self, parent=None):
        """Initialize message input field."""
        super().__init__(parent)
        self.setObjectName("messageInput")
        self.setPlaceholderText("Type your message here...")
        
        # Configure appearance
        self.setMinimumHeight(36)
        
        # Connect signals
        self.returnPressed.connect(self.submit_message)
    
    def submit_message(self):
        """Submit the current message text."""
        message_text = self.text().strip()
        if message_text:
            self.message_submitted.emit(message_text)
            self.clear()


class ConversationPanel(QWidget):
    """Panel for displaying and interacting with conversation."""
    
    message_sent = pyqtSignal(str)
    stop_requested = pyqtSignal()
    new_conversation_requested = pyqtSignal()
    
    def __init__(self, parent=None):
        """Initialize conversation panel."""
        super().__init__(parent)
        self.setObjectName("conversationPanel")
        
        # Initialize conversation state
        self.conversation_state = ConversationState()
        
        # Set up UI
        self.setup_ui()
        
        # Add some initial welcome message
        welcome_message = "Welcome to the conversation! How can I assist you today?"
        self.add_assistant_message(welcome_message)
    
    def setup_ui(self):
        """Set up the conversation panel UI."""
        # Main layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Create scroll area for message history
        self.scroll_area = QScrollArea()
        self.scroll_area.setObjectName("messageScroll")
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        
        # Create container widget for messages
        self.message_container = QWidget()
        self.message_container.setObjectName("messageContainer")
        self.message_layout = QVBoxLayout(self.message_container)
        self.message_layout.setContentsMargins(12, 12, 12, 12)
        self.message_layout.setSpacing(8)
        self.message_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        
        # Add stretch to push messages to the top
        self.message_layout.addStretch()
        
        # Set message container as the scroll area widget
        self.scroll_area.setWidget(self.message_container)
        
        # Create input area
        input_container = QWidget()
        input_container.setObjectName("inputContainer")
        input_layout = QHBoxLayout(input_container)
        input_layout.setContentsMargins(12, 8, 12, 12)
        input_layout.setSpacing(8)
        
        # New conversation button
        self.new_conversation_btn = QPushButton()
        self.new_conversation_btn.setObjectName("newConversationBtn")
        self.new_conversation_btn.setIcon(QIcon())  # Placeholder for icon
        self.new_conversation_btn.setToolTip("New Conversation")
        self.new_conversation_btn.setFixedSize(36, 36)
        self.new_conversation_btn.clicked.connect(self.on_new_conversation)
        
        # Message input field
        self.message_input = MessageInput()
        
        # Send button
        self.send_button = QPushButton()
        self.send_button.setObjectName("sendButton")
        self.send_button.setIcon(QIcon())  # Placeholder for icon
        self.send_button.setToolTip("Send Message")
        self.send_button.setFixedSize(36, 36)
        self.send_button.clicked.connect(self.on_send_clicked)
        
        # Stop button (hidden by default)
        self.stop_button = QPushButton()
        self.stop_button.setObjectName("stopButton")
        self.stop_button.setIcon(QIcon())  # Placeholder for icon
        self.stop_button.setToolTip("Stop Generating")
        self.stop_button.setFixedSize(36, 36)
        self.stop_button.clicked.connect(self.on_stop_clicked)
        self.stop_button.setVisible(False)
        
        # Add widgets to input layout
        input_layout.addWidget(self.new_conversation_btn)
        input_layout.addWidget(self.message_input)
        input_layout.addWidget(self.send_button)
        input_layout.addWidget(self.stop_button)
        
        # Add widgets to main layout
        layout.addWidget(self.scroll_area)
        layout.addWidget(input_container)
        
        # Connect signals
        self.message_input.message_submitted.connect(self.on_message_submitted)
    
    def add_message_bubble(self, message: Message):
        """Add a message bubble to the conversation.
        
        Args:
            message: Message to display
        """
        # Remove the stretch item first
        stretch_item = self.message_layout.takeAt(self.message_layout.count() - 1)
        
        # Create and add the message bubble
        message_bubble = MessageBubble(message, self)
        self.message_layout.addWidget(message_bubble)
        
        # Add the stretch back
        self.message_layout.addItem(stretch_item)
        
        # Scroll to the bottom
        QTimer.singleShot(50, self.scroll_to_bottom)
    
    def scroll_to_bottom(self):
        """Scroll the message area to the bottom."""
        vertical_scrollbar = self.scroll_area.verticalScrollBar()
        vertical_scrollbar.setValue(vertical_scrollbar.maximum())
    
    def add_user_message(self, content: str):
        """Add a user message to the conversation.
        
        Args:
            content: Message content
        """
        message = self.conversation_state.add_user_message(content)
        self.add_message_bubble(message)
    
    def add_assistant_message(self, content: str, status: MessageStatus = MessageStatus.COMPLETE):
        """Add an assistant message to the conversation.
        
        Args:
            content: Message content
            status: Message status
        """
        message = self.conversation_state.add_assistant_message(content, status)
        self.add_message_bubble(message)
    
    def on_message_submitted(self, message_text: str):
        """Handle message submission.
        
        Args:
            message_text: The submitted message text
        """
        # Add user message to conversation
        self.add_user_message(message_text)
        
        # Show the stop button and hide the send button during generation
        self.send_button.setVisible(False)
        self.stop_button.setVisible(True)
        
        # Disable input during generation
        self.message_input.setEnabled(False)
        
        # Emit message_sent signal
        self.message_sent.emit(message_text)
        
        # Simulate a response for demo purposes
        # In a real implementation, this would be handled by the application logic
        QTimer.singleShot(500, lambda: self.simulate_response(message_text))
    
    def on_send_clicked(self):
        """Handle send button click."""
        self.message_input.submit_message()
    
    def on_stop_clicked(self):
        """Handle stop button click."""
        # Hide the stop button and show the send button
        self.stop_button.setVisible(False)
        self.send_button.setVisible(True)
        
        # Enable input
        self.message_input.setEnabled(True)
        
        # Emit stop_requested signal
        self.stop_requested.emit()
    
    def on_new_conversation(self):
        """Handle new conversation button click."""
        # Create a new conversation
        self.conversation_state.new_conversation()
        
        # Clear the message display
        self.clear_messages()
        
        # Add welcome message
        welcome_message = "Started a new conversation. How can I assist you today?"
        self.add_assistant_message(welcome_message)
        
        # Emit new_conversation_requested signal
        self.new_conversation_requested.emit()
    
    def clear_messages(self):
        """Clear all messages from the display."""
        # Remove all message bubbles, keeping the stretch item
        stretch_item = self.message_layout.takeAt(self.message_layout.count() - 1)
        
        while self.message_layout.count() > 0:
            item = self.message_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
        
        # Add the stretch back
        self.message_layout.addItem(stretch_item)
    
    def simulate_response(self, query: str):
        """Simulate an assistant response for demo purposes.
        
        Args:
            query: The user query to respond to
        """
        # Add a "thinking" message
        thinking_message = "Thinking..."
        self.add_assistant_message(thinking_message, MessageStatus.GENERATING)
        
        # Generate a mock response based on the query
        response = self.generate_mock_response(query)
        
        # Simulate typing delay
        QTimer.singleShot(1500, lambda: self.update_response(response))
    
    def update_response(self, response: str):
        """Update the latest assistant message with the full response.
        
        Args:
            response: The full response text
        """
        # Get the latest message (should be the "thinking" message)
        latest_message = self.conversation_state.current_conversation.messages[-1]
        
        # Update the message content
        self.conversation_state.current_conversation.update_message(
            latest_message.id, 
            content=response, 
            status=MessageStatus.COMPLETE
        )
        
        # Clear and rebuild the message display
        self.clear_messages()
        
        # Add all messages back
        for message in self.conversation_state.current_conversation.messages:
            self.add_message_bubble(message)
        
        # Restore the UI state
        self.send_button.setVisible(True)
        self.stop_button.setVisible(False)
        self.message_input.setEnabled(True)
    
    def generate_mock_response(self, query: str) -> str:
        """Generate a mock response for demo purposes.
        
        Args:
            query: The user query to respond to
            
        Returns:
            str: A mock response
        """
        # Simple mock responses based on keywords
        if "hello" in query.lower() or "hi" in query.lower():
            return "Hello there! How can I help you today?"
        
        elif "help" in query.lower():
            return "I'm here to help! You can ask me questions about coding, data analysis, or general information."
        
        elif "code" in query.lower() or "python" in query.lower():
            return """Here's a simple Python example:
            
```python
def greeting(name):
    return f"Hello, {name}!"
    
print(greeting("World"))
```

This will output: `Hello, World!`"""
        
        elif "feature" in query.lower() or "do" in query.lower():
            return """In this application, you can:
            
1. Have conversations with AI
2. Edit code with syntax highlighting
3. Preview web content
4. Run terminal commands
5. Manage projects and files

Is there a specific feature you'd like to know more about?"""
        
        else:
            return "I understand you're asking about: " + query + "\n\nCould you provide more details so I can help you better?"


if __name__ == "__main__":
    # Test code
    import sys
    from PyQt6.QtWidgets import QApplication
    from app.theme import apply_theme
    
    app = QApplication(sys.argv)
    apply_theme(app)
    
    panel = ConversationPanel()
    panel.resize(800, 600)
    panel.show()
    
    sys.exit(app.exec())