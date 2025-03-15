"""
components/web_preview.py
Implements web preview component, sets up navigation controls, and handles content rendering.
"""
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLineEdit, QPushButton, 
    QToolButton, QLabel, QFrame, QSizePolicy, QProgressBar
)
from PyQt6.QtCore import Qt, QUrl, pyqtSignal, pyqtSlot, QSize
from PyQt6.QtGui import QIcon, QAction, QKeySequence
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtWebEngineCore import QWebEnginePage


class NavigationBar(QWidget):
    """Navigation bar for web preview with URL field and controls."""
    
    # Signals
    url_changed = pyqtSignal(str)
    reload_clicked = pyqtSignal()
    back_clicked = pyqtSignal()
    forward_clicked = pyqtSignal()
    stop_clicked = pyqtSignal()
    
    def __init__(self, parent=None):
        """Initialize navigation bar."""
        super().__init__(parent)
        self.setObjectName("navigationBar")
        
        self.setup_ui()
    
    def setup_ui(self):
        """Set up navigation bar UI."""
        # Main layout
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)
        
        # Navigation buttons
        self.back_button = QToolButton()
        self.back_button.setObjectName("backButton")
        self.back_button.setIcon(QIcon())  # Placeholder for back icon
        self.back_button.setIconSize(QSize(16, 16))
        self.back_button.setToolTip("Back")
        self.back_button.clicked.connect(self.back_clicked.emit)
        
        self.forward_button = QToolButton()
        self.forward_button.setObjectName("forwardButton")
        self.forward_button.setIcon(QIcon())  # Placeholder for forward icon
        self.forward_button.setIconSize(QSize(16, 16))
        self.forward_button.setToolTip("Forward")
        self.forward_button.clicked.connect(self.forward_clicked.emit)
        
        self.reload_button = QToolButton()
        self.reload_button.setObjectName("reloadButton")
        self.reload_button.setIcon(QIcon())  # Placeholder for reload icon
        self.reload_button.setIconSize(QSize(16, 16))
        self.reload_button.setToolTip("Reload")
        self.reload_button.clicked.connect(self.reload_clicked.emit)
        
        # URL bar
        self.url_bar = QLineEdit()
        self.url_bar.setObjectName("urlBar")
        self.url_bar.setPlaceholderText("Enter URL")
        self.url_bar.returnPressed.connect(self.on_url_edit_return)
        
        # Add widgets to layout
        layout.addWidget(self.back_button)
        layout.addWidget(self.forward_button)
        layout.addWidget(self.reload_button)
        layout.addWidget(self.url_bar)
    
    def on_url_edit_return(self):
        """Handle return key press in URL bar."""
        url = self.url_bar.text()
        self.url_changed.emit(url)
    
    def set_url(self, url):
        """Set the URL in the URL bar.
        
        Args:
            url: URL to display
        """
        self.url_bar.setText(url)
    
    def set_loading(self, is_loading):
        """Update UI to reflect loading state.
        
        Args:
            is_loading: True if page is loading, False otherwise
        """
        if is_loading:
            # Show stop icon in reload button
            self.reload_button.setToolTip("Stop")
            # Would change icon to stop icon here
            self.reload_button.clicked.disconnect()
            self.reload_button.clicked.connect(self.stop_clicked.emit)
        else:
            # Show reload icon in reload button
            self.reload_button.setToolTip("Reload")
            # Would change icon to reload icon here
            self.reload_button.clicked.disconnect()
            self.reload_button.clicked.connect(self.reload_clicked.emit)
    
    def update_navigation_state(self, can_go_back, can_go_forward):
        """Update navigation button states.
        
        Args:
            can_go_back: True if can navigate back
            can_go_forward: True if can navigate forward
        """
        self.back_button.setEnabled(can_go_back)
        self.forward_button.setEnabled(can_go_forward)


class WebPreview(QWidget):
    """Web preview component for rendering web content."""
    
    def __init__(self, parent=None):
        """Initialize web preview component."""
        super().__init__(parent)
        self.setObjectName("webPreview")
        
        self.setup_ui()
        self.setup_connections()
        
        # Load default page
        self.load_url("http://localhost:3000/")
    
    def setup_ui(self):
        """Set up web preview UI."""
        # Main layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Navigation bar
        self.nav_bar = NavigationBar(self)
        
        # Progress bar
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setObjectName("loadingProgress")
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setMaximumHeight(3)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        
        # Web view
        self.web_view = QWebEngineView(self)
        self.web_view.setObjectName("webEngineView")
        
        # Add widgets to layout
        layout.addWidget(self.nav_bar)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.web_view)
    
    def setup_connections(self):
        """Set up signal/slot connections."""
        # Web view signals
        self.web_view.loadStarted.connect(self.handle_load_started)
        self.web_view.loadProgress.connect(self.handle_load_progress)
        self.web_view.loadFinished.connect(self.handle_load_finished)
        self.web_view.urlChanged.connect(self.handle_url_changed)
        
        # Navigation bar signals
        self.nav_bar.url_changed.connect(self.load_url)
        self.nav_bar.back_clicked.connect(self.web_view.back)
        self.nav_bar.forward_clicked.connect(self.web_view.forward)
        self.nav_bar.reload_clicked.connect(self.web_view.reload)
        self.nav_bar.stop_clicked.connect(self.web_view.stop)
    
    def handle_load_started(self):
        """Handle page load started."""
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        self.nav_bar.set_loading(True)
    
    def handle_load_progress(self, progress):
        """Handle page load progress.
        
        Args:
            progress: Loading progress (0-100)
        """
        self.progress_bar.setValue(progress)
    
    def handle_load_finished(self, success):
        """Handle page load finished.
        
        Args:
            success: True if load was successful
        """
        self.progress_bar.setValue(100)
        self.progress_bar.setVisible(False)
        self.nav_bar.set_loading(False)
        
        # Update navigation state
        self.update_navigation_state()
    
    def handle_url_changed(self, url):
        """Handle URL change.
        
        Args:
            url: New URL
        """
        self.nav_bar.set_url(url.toString())
        self.update_navigation_state()
    
    def update_navigation_state(self):
        """Update navigation button states."""
        self.nav_bar.update_navigation_state(
            self.web_view.page().history().canGoBack(),
            self.web_view.page().history().canGoForward()
        )
    
    def load_url(self, url):
        """Load a URL in the web view.
        
        Args:
            url: URL to load (string or QUrl)
        """
        if isinstance(url, str):
            # Add http:// if no scheme is present
            if not url.startswith(('http://', 'https://', 'file://')):
                url = 'http://' + url
            
            url = QUrl(url)
        
        self.web_view.load(url)
    
    def load_html(self, html, base_url=None):
        """Load HTML content directly.
        
        Args:
            html: HTML content
            base_url: Base URL for resources (optional)
        """
        if base_url is None:
            base_url = QUrl("http://localhost/")
        elif isinstance(base_url, str):
            base_url = QUrl(base_url)
        
        self.web_view.setHtml(html, base_url)
    
    def run_javascript(self, script, callback=None):
        """Run JavaScript in the web page.
        
        Args:
            script: JavaScript code
            callback: Callback function for result (optional)
        """
        if callback:
            self.web_view.page().runJavaScript(script, callback)
        else:
            self.web_view.page().runJavaScript(script)
    
    def toggle_developer_tools(self):
        """Toggle developer tools (Web Inspector)."""
        self.web_view.page().triggerAction(QWebEnginePage.WebAction.InspectElement)


if __name__ == "__main__":
    # Test code
    import sys
    from PyQt6.QtWidgets import QApplication
    from app.theme import apply_theme
    
    app = QApplication(sys.argv)
    apply_theme(app)
    
    preview = WebPreview()
    preview.resize(800, 600)
    preview.show()
    
    sys.exit(app.exec())