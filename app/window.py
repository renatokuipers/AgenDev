"""
app/window.py
Defines MainWindow class, implements main application layout, and sets up theme and styling.
"""
import sys
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QSplitter, QVBoxLayout, QWidget, QHBoxLayout,
    QLabel, QPushButton, QFrame
)
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QIcon, QAction

# Import theme system
from app.theme import apply_theme, get_component_stylesheet

# Import components (these will be implemented later)
# For now, we're just defining the imports based on the knowledge graph
from components.sidebar import Sidebar
from components.content_area import ContentArea
from components.conversation import ConversationPanel


class MainWindow(QMainWindow):
    """Main application window containing all UI components."""
    
    def __init__(self, parent=None):
        """Initialize the main window."""
        super().__init__(parent)
        self.setWindowTitle("same")
        self.resize(1200, 800)
        
        # Configure window properties
        self.setMinimumSize(800, 600)
        self.setWindowIcon(QIcon("path/to/icon.png"))  # Replace with actual icon path
        
        # Set up the main layout
        self.setup_ui()
        
        # Apply style
        self.apply_styles()
    
    def setup_ui(self):
        """Set up the main UI components and layout."""
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main vertical layout
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Create main horizontal splitter (sidebar | content)
        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Create sidebar
        self.sidebar = Sidebar(self)
        self.sidebar.setObjectName("sidebar")
        self.sidebar.setMinimumWidth(200)
        self.sidebar.setMaximumWidth(400)
        
        # Create vertical splitter for content area and conversation panel
        self.content_splitter = QSplitter(Qt.Orientation.Vertical)
        
        # Create content area with tabs
        self.content_area = ContentArea(self)
        self.content_area.setObjectName("contentArea")
        
        # Create conversation panel
        self.conversation_panel = ConversationPanel(self)
        self.conversation_panel.setObjectName("conversationPanel")
        self.conversation_panel.setMinimumHeight(150)
        
        # Add widgets to the content splitter
        self.content_splitter.addWidget(self.content_area)
        self.content_splitter.addWidget(self.conversation_panel)
        
        # Set initial sizes for content_splitter (content area gets more space)
        self.content_splitter.setSizes([600, 200])
        
        # Add widgets to the main splitter
        self.main_splitter.addWidget(self.sidebar)
        self.main_splitter.addWidget(self.content_splitter)
        
        # Set initial sizes for main_splitter (content gets more space)
        self.main_splitter.setSizes([250, 950])
        
        # Add main splitter to the layout
        main_layout.addWidget(self.main_splitter)
        
        # Configure splitter appearance
        for splitter in [self.main_splitter, self.content_splitter]:
            splitter.setHandleWidth(1)
            splitter.setChildrenCollapsible(False)
    
    def apply_styles(self):
        """Apply the styles to all components."""
        # The main stylesheet is applied at the application level
        # Component-specific styles are applied here
        self.sidebar.setStyleSheet(get_component_stylesheet("sidebar"))
        self.content_area.setStyleSheet(get_component_stylesheet("content_area"))
        self.conversation_panel.setStyleSheet(get_component_stylesheet("conversation"))


def create_application():
    """
    Create and configure the QApplication instance.
    
    Returns:
        QApplication: The configured application instance
    """
    app = QApplication(sys.argv)
    
    # Apply dark theme
    apply_theme(app)
    
    return app


def run_application():
    """
    Create and run the main application window.
    """
    app = create_application()
    
    # Create and show the main window
    window = MainWindow()
    window.show()
    
    # Start the application event loop
    sys.exit(app.exec())


if __name__ == "__main__":
    run_application()