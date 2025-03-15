"""
components/sidebar.py
Implements Sidebar component, creates project tree widget, and sets up sidebar layout and controls.
"""
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QTreeWidget, QTreeWidgetItem, QPushButton, 
    QLabel, QHBoxLayout, QFrame, QSizePolicy, QSpacerItem
)
from PyQt6.QtCore import Qt, QSize, pyqtSignal
from PyQt6.QtGui import QIcon, QFont, QColor

# These will be implemented later
from models.project import Project
from utils.file_operations import get_recent_projects


class ProjectTreeItem(QTreeWidgetItem):
    """Custom tree item for projects and files."""
    
    def __init__(self, parent=None, item_type="file", name="", path=""):
        """Initialize the tree item.
        
        Args:
            parent: Parent item or tree
            item_type: Type of item ("project", "folder", "file", etc.)
            name: Display name
            path: File path
        """
        super().__init__(parent)
        self.item_type = item_type
        self.name = name
        self.path = path
        
        self.setText(0, name)
        self.setData(0, Qt.ItemDataRole.UserRole, path)
        
        # Set icon based on type (would use actual icons in production)
        if item_type == "project":
            self.setIcon(0, QIcon())  # Project icon placeholder
        elif item_type == "folder":
            self.setIcon(0, QIcon())  # Folder icon placeholder
        elif item_type == "file":
            self.setIcon(0, QIcon())  # File icon placeholder
        
        # Customize appearance
        if item_type == "project":
            font = self.font(0)
            font.setBold(True)
            self.setFont(0, font)


class CollapsibleSection(QWidget):
    """A collapsible section widget for the sidebar."""
    
    def __init__(self, title, parent=None):
        """Initialize collapsible section.
        
        Args:
            title: Section title
            parent: Parent widget
        """
        super().__init__(parent)
        self.title = title
        self.is_collapsed = False
        
        self.setup_ui()
    
    def setup_ui(self):
        """Set up the section UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Header with title and toggle button
        header = QWidget()
        header.setObjectName("sectionHeader")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(8, 4, 8, 4)
        
        # Section title
        title_label = QLabel(self.title)
        title_label.setObjectName("sectionTitle")
        
        # Toggle indicator (▼/▶)
        self.toggle_btn = QPushButton("▼")
        self.toggle_btn.setObjectName("sectionToggle")
        self.toggle_btn.setFixedSize(16, 16)
        self.toggle_btn.clicked.connect(self.toggle_section)
        
        header_layout.addWidget(title_label)
        header_layout.addStretch()
        header_layout.addWidget(self.toggle_btn)
        
        # Content container
        self.content = QWidget()
        self.content_layout = QVBoxLayout(self.content)
        self.content_layout.setContentsMargins(0, 0, 0, 0)
        self.content_layout.setSpacing(0)
        
        # Add widgets to main layout
        layout.addWidget(header)
        layout.addWidget(self.content)
    
    def add_widget(self, widget):
        """Add a widget to the section content.
        
        Args:
            widget: Widget to add
        """
        self.content_layout.addWidget(widget)
    
    def toggle_section(self):
        """Toggle the section between collapsed and expanded."""
        self.is_collapsed = not self.is_collapsed
        self.content.setVisible(not self.is_collapsed)
        
        # Update toggle button text
        if self.is_collapsed:
            self.toggle_btn.setText("▶")
        else:
            self.toggle_btn.setText("▼")


class Sidebar(QWidget):
    """Sidebar component with project navigation."""
    
    # Signals
    project_selected = pyqtSignal(str)  # Emits project path
    file_selected = pyqtSignal(str)     # Emits file path
    new_project_clicked = pyqtSignal()  # Emits when new project button clicked
    
    def __init__(self, parent=None):
        """Initialize sidebar component."""
        super().__init__(parent)
        self.setObjectName("sidebar")
        
        # Limit width
        self.setMinimumWidth(200)
        self.setMaximumWidth(400)
        
        self.setup_ui()
        self.load_recent_projects()
    
    def setup_ui(self):
        """Set up the sidebar UI components."""
        # Main layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # New Project button
        new_project_btn = QPushButton("+ New Project")
        new_project_btn.setObjectName("newProjectBtn")
        new_project_btn.clicked.connect(self.on_new_project_clicked)
        
        # Button container
        btn_container = QWidget()
        btn_layout = QVBoxLayout(btn_container)
        btn_layout.setContentsMargins(10, 10, 10, 10)
        btn_layout.addWidget(new_project_btn)
        
        # "Yesterday" section
        self.yesterday_section = CollapsibleSection("Yesterday")
        
        # Project section
        self.tree_widget = QTreeWidget()
        self.tree_widget.setObjectName("projectTree")
        self.tree_widget.setHeaderHidden(True)
        self.tree_widget.setIndentation(15)
        self.tree_widget.setAnimated(True)
        self.tree_widget.itemClicked.connect(self.on_tree_item_clicked)
        
        # Add tree to yesterday section
        self.yesterday_section.add_widget(self.tree_widget)
        
        # "All projects" button
        all_projects_btn = QPushButton("···  All projects")
        all_projects_btn.setObjectName("allProjectsBtn")
        all_projects_btn.setIcon(QIcon())  # Placeholder for icon
        
        # Feedback section
        feedback_label = QLabel("Feedback?")
        feedback_label.setObjectName("feedbackLabel")
        
        social_container = QWidget()
        social_layout = QVBoxLayout(social_container)
        social_layout.setContentsMargins(10, 5, 10, 5)
        social_layout.setSpacing(8)
        
        # Social buttons (Twitter, Slack)
        twitter_btn = QPushButton("@samedotdev")
        twitter_btn.setObjectName("twitterBtn")
        twitter_btn.setIcon(QIcon())  # Placeholder for Twitter icon
        
        slack_btn = QPushButton("Slack channel")
        slack_btn.setObjectName("slackBtn")
        slack_btn.setIcon(QIcon())  # Placeholder for Slack icon
        
        social_layout.addWidget(twitter_btn)
        social_layout.addWidget(slack_btn)
        
        # User profile section at bottom
        user_container = QWidget()
        user_container.setObjectName("userProfile")
        user_layout = QHBoxLayout(user_container)
        
        # User avatar placeholder
        user_avatar = QLabel()
        user_avatar.setFixedSize(32, 32)
        user_avatar.setObjectName("userAvatar")
        
        # User info
        user_info = QWidget()
        user_info_layout = QVBoxLayout(user_info)
        user_info_layout.setContentsMargins(0, 0, 0, 0)
        user_info_layout.setSpacing(0)
        
        user_name = QLabel("Ren Jestoo")
        user_name.setObjectName("userName")
        user_email = QLabel("renjestoo@gmail.com")
        user_email.setObjectName("userEmail")
        user_email.setStyleSheet("color: rgba(255, 255, 255, 0.6);")
        
        user_info_layout.addWidget(user_name)
        user_info_layout.addWidget(user_email)
        
        user_layout.addWidget(user_avatar)
        user_layout.addWidget(user_info)
        user_layout.addStretch()
        
        # Divider between main content and feedback section
        divider = QFrame()
        divider.setFrameShape(QFrame.Shape.HLine)
        divider.setFrameShadow(QFrame.Shadow.Sunken)
        divider.setObjectName("sidebarDivider")
        
        # Add all widgets to main layout
        layout.addWidget(btn_container)
        layout.addWidget(self.yesterday_section)
        layout.addWidget(all_projects_btn)
        layout.addStretch(1)
        layout.addWidget(divider)
        layout.addWidget(feedback_label)
        layout.addWidget(social_container)
        layout.addWidget(user_container)
    
    def load_recent_projects(self):
        """Load recent projects into the tree widget."""
        try:
            # This would normally get data from models/project.py
            # For now we'll create sample projects
            sample_projects = [
                {"name": "Researching /opt/same", "path": "/opt/same"},
                {"name": "No File Attached", "path": ""},
                {"name": "Combine UI Screenshots", "path": "/projects/ui_screenshots"},
                {"name": "Clone UI", "path": "/projects/clone_ui"},
                {"name": "Clone https://manus.im/same", "path": "https://manus.im/same"}
            ]
            
            self.tree_widget.clear()
            
            for project in sample_projects:
                project_item = ProjectTreeItem(
                    self.tree_widget,
                    item_type="project",
                    name=project["name"],
                    path=project["path"]
                )
                
                # For demo purpose, add some child items to first project
                if project == sample_projects[0]:
                    for i in range(3):
                        file_item = ProjectTreeItem(
                            project_item,
                            item_type="file",
                            name=f"File {i+1}.py",
                            path=f"{project['path']}/file_{i+1}.py"
                        )
            
            # Expand the first project
            first_item = self.tree_widget.topLevelItem(0)
            if first_item:
                self.tree_widget.expandItem(first_item)
                
        except Exception as e:
            print(f"Error loading recent projects: {e}")
    
    def on_tree_item_clicked(self, item, column):
        """Handle tree item click event.
        
        Args:
            item: The clicked QTreeWidgetItem
            column: Column index
        """
        item_type = item.item_type
        path = item.path
        
        if item_type == "project":
            self.project_selected.emit(path)
        elif item_type == "file":
            self.file_selected.emit(path)
    
    def on_new_project_clicked(self):
        """Handle new project button click."""
        self.new_project_clicked.emit()


if __name__ == "__main__":
    # Test code
    import sys
    from PyQt6.QtWidgets import QApplication
    from app.theme import apply_theme
    
    app = QApplication(sys.argv)
    apply_theme(app)
    
    sidebar = Sidebar()
    sidebar.show()
    
    sys.exit(app.exec())