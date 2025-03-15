"""
app/theme.py
Defines color constants, stylesheets, and theme utility functions for the application.
"""
from PyQt6.QtGui import QColor, QPalette, QFont
from PyQt6.QtWidgets import QApplication


# Color constants
class Colors:
    # Primary colors
    BACKGROUND = "#121212"
    SURFACE = "#1E1E1E"
    SURFACE_LIGHT = "#252525"
    SIDEBAR_BG = "#0F0F0F"
    
    # Text colors
    TEXT_PRIMARY = "#FFFFFF"
    TEXT_SECONDARY = "#AAAAAA"
    TEXT_DISABLED = "#666666"
    
    # Accent colors
    ACCENT_PRIMARY = "#4285F4"  # Blue accent color
    ACCENT_SUCCESS = "#0F9D58"  # Green for success indicators
    ACCENT_WARNING = "#F4B400"  # Yellow for warnings
    ACCENT_ERROR = "#DB4437"    # Red for errors
    
    # Specialized UI element colors
    BORDER = "#333333"
    HOVER = "#2A2A2A"
    SELECTED = "#383838"
    DIVIDER = "#333333"
    
    # Terminal colors
    TERMINAL_BG = "#0A0A0A"
    TERMINAL_TEXT = "#C0C0C0"
    TERMINAL_CURSOR = "#FFFFFF"
    
    # Code editor colors
    CODE_EDITOR_BG = "#1A1A1A"
    CODE_EDITOR_LINE_NUMBER_BG = "#151515"
    CODE_EDITOR_LINE_NUMBER_FG = "#666666"
    CODE_EDITOR_SELECTION = "#153F65"


# Font settings
class Fonts:
    FAMILY_DEFAULT = "Segoe UI"  # Good default for Windows
    FAMILY_MONOSPACE = "Consolas"
    
    SIZE_SMALL = 9
    SIZE_NORMAL = 10
    SIZE_MEDIUM = 11
    SIZE_LARGE = 13
    SIZE_XLARGE = 15


# Stylesheet definitions
class Stylesheets:
    BASE = f"""
    QWidget {{
        background-color: {Colors.BACKGROUND};
        color: {Colors.TEXT_PRIMARY};
        font-family: '{Fonts.FAMILY_DEFAULT}';
        font-size: {Fonts.SIZE_NORMAL}pt;
    }}
    
    QMainWindow {{
        background-color: {Colors.BACKGROUND};
        border: none;
    }}
    
    QSplitter::handle {{
        background-color: {Colors.DIVIDER};
        height: 1px;
        width: 1px;
    }}
    
    QScrollBar:vertical {{
        border: none;
        background: {Colors.SURFACE};
        width: 10px;
        margin: 0px;
    }}
    
    QScrollBar::handle:vertical {{
        background: {Colors.SELECTED};
        min-height: 20px;
        border-radius: 5px;
    }}
    
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
        height: 0px;
    }}
    
    QScrollBar:horizontal {{
        border: none;
        background: {Colors.SURFACE};
        height: 10px;
        margin: 0px;
    }}
    
    QScrollBar::handle:horizontal {{
        background: {Colors.SELECTED};
        min-width: 20px;
        border-radius: 5px;
    }}
    
    QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
        width: 0px;
    }}
    
    QPushButton {{
        background-color: {Colors.SURFACE_LIGHT};
        border: none;
        border-radius: 4px;
        padding: 5px 10px;
        color: {Colors.TEXT_PRIMARY};
    }}
    
    QPushButton:hover {{
        background-color: {Colors.HOVER};
    }}
    
    QPushButton:pressed {{
        background-color: {Colors.SELECTED};
    }}
    
    QPushButton:disabled {{
        background-color: {Colors.SURFACE};
        color: {Colors.TEXT_DISABLED};
    }}
    
    QLineEdit {{
        background-color: {Colors.SURFACE};
        border: 1px solid {Colors.BORDER};
        border-radius: 4px;
        padding: 5px;
        color: {Colors.TEXT_PRIMARY};
    }}
    
    QLineEdit:focus {{
        border: 1px solid {Colors.ACCENT_PRIMARY};
    }}
    
    QTabWidget::pane {{
        border: 1px solid {Colors.BORDER};
        background-color: {Colors.SURFACE};
    }}
    
    QTabBar::tab {{
        background-color: {Colors.SURFACE};
        color: {Colors.TEXT_SECONDARY};
        padding: 8px 12px;
        border: none;
    }}
    
    QTabBar::tab:selected {{
        background-color: {Colors.SURFACE_LIGHT};
        color: {Colors.TEXT_PRIMARY};
        border-bottom: 2px solid {Colors.ACCENT_PRIMARY};
    }}
    
    QTabBar::tab:hover:!selected {{
        background-color: {Colors.HOVER};
    }}
    """
    
    SIDEBAR = f"""
    QWidget#sidebar {{
        background-color: {Colors.SIDEBAR_BG};
        border-right: 1px solid {Colors.BORDER};
    }}
    
    QTreeWidget {{
        background-color: {Colors.SIDEBAR_BG};
        border: none;
        outline: none;
        padding: 5px;
    }}
    
    QTreeWidget::item {{
        padding: 5px;
        border-radius: 4px;
    }}
    
    QTreeWidget::item:selected {{
        background-color: {Colors.SELECTED};
    }}
    
    QTreeWidget::item:hover:!selected {{
        background-color: {Colors.HOVER};
    }}
    """
    
    CONTENT_AREA = f"""
    QTabWidget#contentTabs {{
        background-color: {Colors.SURFACE};
    }}
    """
    
    CODE_EDITOR = f"""
    QsciScintilla {{
        background-color: {Colors.CODE_EDITOR_BG};
        color: {Colors.TEXT_PRIMARY};
        border: none;
        font-family: '{Fonts.FAMILY_MONOSPACE}';
        font-size: {Fonts.SIZE_NORMAL}pt;
    }}
    """
    
    TERMINAL = f"""
    QWidget#terminal {{
        background-color: {Colors.TERMINAL_BG};
        color: {Colors.TERMINAL_TEXT};
        font-family: '{Fonts.FAMILY_MONOSPACE}';
        font-size: {Fonts.SIZE_NORMAL}pt;
        border: none;
    }}
    """
    
    CONVERSATION = f"""
    QWidget#conversationPanel {{
        background-color: {Colors.SURFACE};
        border-top: 1px solid {Colors.BORDER};
    }}
    
    QTextEdit#messageHistory {{
        background-color: {Colors.SURFACE};
        border: none;
        font-size: {Fonts.SIZE_NORMAL}pt;
    }}
    
    QLineEdit#messageInput {{
        background-color: {Colors.SURFACE_LIGHT};
        border: 1px solid {Colors.BORDER};
        border-radius: 4px;
        padding: 8px;
        font-size: {Fonts.SIZE_NORMAL}pt;
    }}
    
    QPushButton#sendButton {{
        background-color: {Colors.ACCENT_PRIMARY};
        color: {Colors.TEXT_PRIMARY};
        border-radius: 4px;
        padding: 8px;
    }}
    
    QPushButton#sendButton:hover {{
        background-color: #5294FF;
    }}
    
    QPushButton#stopButton {{
        background-color: {Colors.ACCENT_ERROR};
        color: {Colors.TEXT_PRIMARY};
        border-radius: 4px;
        padding: 8px;
    }}
    """
    
    WEB_PREVIEW = f"""
    QWidget#webPreview {{
        background-color: {Colors.SURFACE};
    }}
    
    QLineEdit#urlBar {{
        background-color: {Colors.SURFACE_LIGHT};
        border: 1px solid {Colors.BORDER};
        border-radius: 4px;
        padding: 5px;
        font-size: {Fonts.SIZE_NORMAL}pt;
    }}
    """


def apply_theme(app: QApplication) -> None:
    """
    Apply the dark theme to the entire application.
    
    Args:
        app: The QApplication instance
    """
    # Apply the stylesheet
    app.setStyleSheet(Stylesheets.BASE)
    
    # Set application-wide palette
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(Colors.BACKGROUND))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(Colors.TEXT_PRIMARY))
    palette.setColor(QPalette.ColorRole.Base, QColor(Colors.SURFACE))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(Colors.SURFACE_LIGHT))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(Colors.SURFACE))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor(Colors.TEXT_PRIMARY))
    palette.setColor(QPalette.ColorRole.Text, QColor(Colors.TEXT_PRIMARY))
    palette.setColor(QPalette.ColorRole.Button, QColor(Colors.SURFACE_LIGHT))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(Colors.TEXT_PRIMARY))
    palette.setColor(QPalette.ColorRole.BrightText, QColor(Colors.TEXT_PRIMARY))
    palette.setColor(QPalette.ColorRole.Link, QColor(Colors.ACCENT_PRIMARY))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(Colors.ACCENT_PRIMARY))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(Colors.TEXT_PRIMARY))
    
    app.setPalette(palette)


def get_component_stylesheet(component_name: str) -> str:
    """
    Get the stylesheet for a specific component.
    
    Args:
        component_name: Name of the component (sidebar, content_area, etc.)
        
    Returns:
        str: The component-specific stylesheet
    """
    component_name = component_name.lower()
    if component_name == "sidebar":
        return Stylesheets.SIDEBAR
    elif component_name == "content_area":
        return Stylesheets.CONTENT_AREA
    elif component_name == "code_editor":
        return Stylesheets.CODE_EDITOR
    elif component_name == "terminal":
        return Stylesheets.TERMINAL
    elif component_name == "conversation":
        return Stylesheets.CONVERSATION
    elif component_name == "web_preview":
        return Stylesheets.WEB_PREVIEW
    else:
        return ""