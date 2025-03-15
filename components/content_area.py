"""
components/content_area.py
Implements tabbed content area, manages tab switching, and contains tab controls and indicators.
"""
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QTabWidget, QTabBar, QStackedWidget, QFrame, QToolButton,
    QSizePolicy
)
from PyQt6.QtCore import Qt, QSize, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QIcon, QFont

# These will be implemented later
from components.code_editor import CodeEditor
from components.web_preview import WebPreview
from components.terminal import Terminal


class TabButton(QToolButton):
    """Custom button for tab bar with hover effects."""
    
    def __init__(self, icon_name, tooltip, parent=None):
        """Initialize custom tab button.
        
        Args:
            icon_name: Icon name/path
            tooltip: Button tooltip text
            parent: Parent widget
        """
        super().__init__(parent)
        self.setToolTip(tooltip)
        self.setIcon(QIcon(icon_name))  # Placeholder for now
        self.setIconSize(QSize(16, 16))
        self.setFixedSize(24, 24)
        self.setStyleSheet("""
            QToolButton {
                background: transparent;
                border: none;
                border-radius: 4px;
            }
            QToolButton:hover {
                background: rgba(255, 255, 255, 0.1);
            }
            QToolButton:pressed {
                background: rgba(255, 255, 255, 0.2);
            }
        """)


class CustomTabBar(QTabBar):
    """Custom tab bar with improved styling and behavior."""
    
    def __init__(self, parent=None):
        """Initialize custom tab bar."""
        super().__init__(parent)
        self.setDrawBase(False)
        self.setExpanding(False)
        self.setMovable(True)
        
        # Use document mode for cleaner look (flatter tabs)
        self.setDocumentMode(True)
        
        # Custom context menu will be added later


class ContentTabs(QTabWidget):
    """Tab widget to manage code, preview, and terminal tabs."""
    
    # Signals
    tab_changed = pyqtSignal(int)
    
    def __init__(self, parent=None):
        """Initialize content tabs widget."""
        super().__init__(parent)
        self.setObjectName("contentTabs")
        
        # Set up custom tab bar
        self.setTabBar(CustomTabBar())
        
        # Configure tab widget
        self.setTabPosition(QTabWidget.TabPosition.North)
        self.setMovable(True)
        self.setDocumentMode(True)
        
        # Connect signals
        self.currentChanged.connect(self.on_tab_changed)
    
    def on_tab_changed(self, index):
        """Handle tab change event.
        
        Args:
            index: Index of the newly selected tab
        """
        self.tab_changed.emit(index)


class ContentArea(QWidget):
    """Content area component with tabbed interface."""
    
    # Signals
    file_saved = pyqtSignal(str)  # Emits path of saved file
    
    def __init__(self, parent=None):
        """Initialize content area component."""
        super().__init__(parent)
        self.setObjectName("contentArea")
        
        self.active_tab = 0  # Track active tab
        
        self.setup_ui()
    
    def setup_ui(self):
        """Set up the content area UI components."""
        # Main layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Create top tab bar with buttons (App, Code, Terminal)
        self.tab_control = QWidget()
        self.tab_control.setObjectName("tabControl")
        tab_control_layout = QHBoxLayout(self.tab_control)
        tab_control_layout.setContentsMargins(8, 4, 8, 0)
        
        # Tab buttons
        self.app_button = QPushButton("App")
        self.app_button.setObjectName("appTabBtn")
        self.app_button.setCheckable(True)
        
        self.code_button = QPushButton("Code")
        self.code_button.setObjectName("codeTabBtn")
        self.code_button.setCheckable(True)
        
        self.terminal_button = QPushButton("Terminal")
        self.terminal_button.setObjectName("terminalTabBtn")
        self.terminal_button.setCheckable(True)
        
        # Select default tab
        self.code_button.setChecked(True)
        
        # Connect button signals
        self.app_button.clicked.connect(lambda: self.switch_tab(0))
        self.code_button.clicked.connect(lambda: self.switch_tab(1))
        self.terminal_button.clicked.connect(lambda: self.switch_tab(2))
        
        # Add spacer to push buttons to the left
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        
        # Add buttons to layout
        tab_control_layout.addWidget(self.app_button)
        tab_control_layout.addWidget(self.code_button)
        tab_control_layout.addWidget(self.terminal_button)
        tab_control_layout.addWidget(spacer)
        
        # Create stacked widget for content
        self.content_stack = QStackedWidget()
        
        # Create content components
        self.web_preview = WebPreview(self)
        self.code_editor = CodeEditor(self)
        self.terminal = Terminal(self)
        
        # Add widgets to stack
        self.content_stack.addWidget(self.web_preview)
        self.content_stack.addWidget(self.code_editor)
        self.content_stack.addWidget(self.terminal)
        
        # Subtabs for code editor
        self.code_tabs = ContentTabs()
        self.code_tabs.setVisible(False)  # Hide initially, show when code tab is selected
        
        # Add default tabs to code editor tab widget
        # In a full implementation, these would be dynamically created and managed
        self.add_code_tab("layout.tsx", "path/to/layout.tsx")
        self.add_code_tab("page.tsx", "path/to/page.tsx")
        
        # Add widgets to main layout
        layout.addWidget(self.tab_control)
        layout.addWidget(self.code_tabs)
        layout.addWidget(self.content_stack)
        
        # Set initial active tab
        self.switch_tab(1)  # Start with code editor
    
    def switch_tab(self, index):
        """Switch to the specified tab.
        
        Args:
            index: Tab index to switch to
        """
        self.content_stack.setCurrentIndex(index)
        self.active_tab = index
        
        # Update button states
        self.app_button.setChecked(index == 0)
        self.code_button.setChecked(index == 1)
        self.terminal_button.setChecked(index == 2)
        
        # Show/hide code tabs depending on active tab
        self.code_tabs.setVisible(index == 1)
    
    def add_code_tab(self, name, file_path):
        """Add a new tab to the code editor.
        
        Args:
            name: Display name for the tab
            file_path: Path to the file
        """
        # Create a simple placeholder widget for now
        # In a real implementation, this would be a code editor instance
        tab_content = QWidget()
        tab_layout = QVBoxLayout(tab_content)
        tab_layout.addWidget(QLabel(f"Content for {name}"))
        
        # Add tab to the tab widget
        self.code_tabs.addTab(tab_content, name)
    
    @pyqtSlot(str)
    def open_file(self, file_path):
        """Open a file in the appropriate editor.
        
        Args:
            file_path: Path to the file to open
        """
        # Determine file type and open in appropriate tab
        # For now, just add a new code tab as an example
        file_name = file_path.split("/")[-1]
        self.add_code_tab(file_name, file_path)
        
        # Switch to code tab
        self.switch_tab(1)


if __name__ == "__main__":
    # Test code
    import sys
    from PyQt6.QtWidgets import QApplication
    from app.theme import apply_theme
    
    app = QApplication(sys.argv)
    apply_theme(app)
    
    content_area = ContentArea()
    content_area.resize(800, 600)
    content_area.show()
    
    sys.exit(app.exec())