"""
components/code_editor.py
Implements code editor component, sets up syntax highlighting, and configures editor behavior.
"""
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFrame, QLabel, 
    QSplitter, QFileDialog, QMessageBox, QMenu, QApplication
)
from PyQt6.QtCore import Qt, QSize, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QColor, QFont, QFontMetrics, QKeySequence, QShortcut

# Import QScintilla
from PyQt6.Qsci import QsciScintilla, QsciLexerPython, QsciLexerJavaScript, QsciLexerHTML, QsciLexerCSS


class CodeEditor(QWidget):
    """Code editor component with syntax highlighting and advanced editing features."""
    
    # Signals
    file_saved = pyqtSignal(str)  # Emits path of saved file
    content_changed = pyqtSignal()  # Emits when content changes
    
    def __init__(self, parent=None):
        """Initialize code editor component."""
        super().__init__(parent)
        self.setObjectName("codeEditor")
        
        # Current file data
        self.current_file_path = None
        self.current_language = None
        
        self.setup_ui()
        self.setup_shortcuts()
    
    def setup_ui(self):
        """Set up the code editor UI components."""
        # Main layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Create Scintilla editor
        self.editor = QsciScintilla()
        self.editor.setObjectName("scintillaEditor")
        
        # Set up editor appearance
        self.configure_editor()
        
        # Default to Python syntax highlighting for now
        self.set_language("python")
        
        # Add editor to layout
        layout.addWidget(self.editor)
    
    def configure_editor(self):
        """Configure editor appearance and behavior."""
        # Font configuration
        font = QFont("Consolas", 10)
        font.setFixedPitch(True)
        self.editor.setFont(font)
        self.editor.setMarginsFont(font)
        
        # Line numbers margin
        fontmetrics = QFontMetrics(font)
        self.editor.setMarginWidth(0, fontmetrics.horizontalAdvance("00000") + 6)
        self.editor.setMarginLineNumbers(0, True)
        
        # Set margin background color
        self.editor.setMarginsBackgroundColor(QColor("#1A1A1A"))
        self.editor.setMarginsForegroundColor(QColor("#666666"))
        
        # Brace matching
        self.editor.setBraceMatching(QsciScintilla.BraceMatch.SloppyBraceMatch)
        
        # Auto indentation
        self.editor.setAutoIndent(True)
        self.editor.setIndentationGuides(True)
        self.editor.setIndentationsUseTabs(False)
        self.editor.setTabWidth(4)
        
        # Set up folding
        self.editor.setFolding(QsciScintilla.FoldStyle.BoxedTreeFoldStyle)
        self.editor.setFoldMarginColors(QColor("#1A1A1A"), QColor("#1A1A1A"))
        
        # Current line highlight
        self.editor.setCaretLineVisible(True)
        self.editor.setCaretLineBackgroundColor(QColor("#2A2A2A"))
        self.editor.setCaretForegroundColor(QColor("#FFFFFF"))
        
        # Selection color
        self.editor.setSelectionBackgroundColor(QColor("#153F65"))
        self.editor.setSelectionForegroundColor(QColor("#FFFFFF"))
        
        # Set up autocompletion (basic)
        self.editor.setAutoCompletionThreshold(2)
        self.editor.setAutoCompletionSource(QsciScintilla.AutoCompletionSource.AcsAll)
        
        # Connection for content changes
        self.editor.textChanged.connect(self.on_content_changed)
    
    def setup_shortcuts(self):
        """Set up keyboard shortcuts."""
        # Save
        save_shortcut = QShortcut(QKeySequence("Ctrl+S"), self)
        save_shortcut.activated.connect(self.save_file)
        
        # Open
        open_shortcut = QShortcut(QKeySequence("Ctrl+O"), self)
        open_shortcut.activated.connect(self.open_file_dialog)
        
        # Find
        find_shortcut = QShortcut(QKeySequence("Ctrl+F"), self)
        find_shortcut.activated.connect(self.find_text)
        
        # Replace
        replace_shortcut = QShortcut(QKeySequence("Ctrl+H"), self)
        replace_shortcut.activated.connect(self.replace_text)
    
    def set_language(self, language):
        """Set the syntax highlighting language.
        
        Args:
            language: Language identifier (python, javascript, etc.)
        """
        self.current_language = language.lower()
        
        # Clear any existing lexer
        self.editor.setLexer(None)
        
        lexer = None
        
        # Create appropriate lexer based on language
        if language == "python":
            lexer = QsciLexerPython()
        elif language in ["javascript", "js", "jsx", "ts", "tsx"]:
            lexer = QsciLexerJavaScript()
        elif language in ["html", "xml"]:
            lexer = QsciLexerHTML()
        elif language == "css":
            lexer = QsciLexerCSS()
        
        # Apply lexer if one was created
        if lexer:
            # Set lexer font
            lexer.setFont(self.editor.font())
            
            # Update lexer colors for dark theme
            self.configure_lexer_colors(lexer)
            
            # Apply the lexer
            self.editor.setLexer(lexer)
    
    def configure_lexer_colors(self, lexer):
        """Configure lexer colors for dark theme.
        
        Args:
            lexer: The QsciLexer instance to configure
        """
        # Set paper (background) color for all styles
        lexer.setPaper(QColor("#1A1A1A"))
        
        # Set default text color
        lexer.setColor(QColor("#FFFFFF"))
        
        # Python-specific style colors
        if isinstance(lexer, QsciLexerPython):
            lexer.setColor(QColor("#569CD6"), QsciLexerPython.Keyword)  # Keywords
            lexer.setColor(QColor("#CE9178"), QsciLexerPython.DoubleQuotedString)  # Strings
            lexer.setColor(QColor("#CE9178"), QsciLexerPython.SingleQuotedString)  # Strings
            lexer.setColor(QColor("#B5CEA8"), QsciLexerPython.Number)  # Numbers
            lexer.setColor(QColor("#6A9955"), QsciLexerPython.Comment)  # Comments
            lexer.setColor(QColor("#6A9955"), QsciLexerPython.CommentBlock)  # Block comments
            lexer.setColor(QColor("#4EC9B0"), QsciLexerPython.ClassName)  # Class names
            lexer.setColor(QColor("#DCDCAA"), QsciLexerPython.FunctionMethodName)  # Function names
        
        # JavaScript-specific style colors
        elif isinstance(lexer, QsciLexerJavaScript):
            lexer.setColor(QColor("#569CD6"), QsciLexerJavaScript.Keyword)  # Keywords
            lexer.setColor(QColor("#CE9178"), QsciLexerJavaScript.DoubleQuotedString)  # Strings
            lexer.setColor(QColor("#CE9178"), QsciLexerJavaScript.SingleQuotedString)  # Strings
            lexer.setColor(QColor("#B5CEA8"), QsciLexerJavaScript.Number)  # Numbers
            lexer.setColor(QColor("#6A9955"), QsciLexerJavaScript.Comment)  # Comments
            lexer.setColor(QColor("#6A9955"), QsciLexerJavaScript.CommentLine)  # Line comments
            lexer.setColor(QColor("#6A9955"), QsciLexerJavaScript.CommentDoc)  # Doc comments
        
        # HTML-specific style colors
        elif isinstance(lexer, QsciLexerHTML):
            lexer.setColor(QColor("#569CD6"), QsciLexerHTML.Tag)  # Tags
            lexer.setColor(QColor("#9CDCFE"), QsciLexerHTML.Attribute)  # Attributes
            lexer.setColor(QColor("#CE9178"), QsciLexerHTML.HTMLDoubleQuotedString)  # Strings
            lexer.setColor(QColor("#CE9178"), QsciLexerHTML.HTMLSingleQuotedString)  # Strings
            lexer.setColor(QColor("#6A9955"), QsciLexerHTML.HTMLComment)  # Comments
        
        # CSS-specific style colors
        elif isinstance(lexer, QsciLexerCSS):
            lexer.setColor(QColor("#9CDCFE"), QsciLexerCSS.Selector)  # Selectors
            lexer.setColor(QColor("#CE9178"), QsciLexerCSS.DoubleQuotedString)  # Strings
            lexer.setColor(QColor("#CE9178"), QsciLexerCSS.SingleQuotedString)  # Strings
            lexer.setColor(QColor("#B5CEA8"), QsciLexerCSS.CSS1Property)  # Properties
            lexer.setColor(QColor("#B5CEA8"), QsciLexerCSS.CSS2Property)  # CSS2 Properties
            lexer.setColor(QColor("#B5CEA8"), QsciLexerCSS.CSS3Property)  # CSS3 Properties
            lexer.setColor(QColor("#6A9955"), QsciLexerCSS.Comment)  # Comments
    
    def detect_language_from_file(self, file_path):
        """Detect language based on file extension.
        
        Args:
            file_path: Path to the file
            
        Returns:
            str: Detected language identifier
        """
        extension = file_path.split(".")[-1].lower()
        
        # Map extensions to languages
        extension_map = {
            "py": "python",
            "js": "javascript",
            "jsx": "javascript",
            "ts": "javascript",
            "tsx": "javascript",
            "html": "html",
            "htm": "html",
            "xml": "html",
            "css": "css"
        }
        
        return extension_map.get(extension, "python")  # Default to Python
    
    def load_file(self, file_path):
        """Load file content into the editor.
        
        Args:
            file_path: Path to the file to load
        """
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read()
                self.editor.setText(content)
                
                # Detect and set language
                language = self.detect_language_from_file(file_path)
                self.set_language(language)
                
                # Update current file path
                self.current_file_path = file_path
                
                # Reset modification state
                self.editor.setModified(False)
                
                # Move cursor to start
                self.editor.setCursorPosition(0, 0)
                
                return True
        except Exception as e:
            QMessageBox.critical(self, "Error Opening File", f"Could not open file: {str(e)}")
            return False
    
    def save_file(self, path=None):
        """Save the current content to a file.
        
        Args:
            path: File path to save to, if None uses current file path
        
        Returns:
            bool: True if saved successfully
        """
        save_path = path or self.current_file_path
        
        # If no path is set, prompt for one
        if not save_path:
            return self.save_file_as()
        
        try:
            with open(save_path, "w", encoding="utf-8") as file:
                file.write(self.editor.text())
                
                # Update state
                self.editor.setModified(False)
                self.current_file_path = save_path
                
                # Emit signal
                self.file_saved.emit(save_path)
                
                return True
        except Exception as e:
            QMessageBox.critical(self, "Error Saving File", f"Could not save file: {str(e)}")
            return False
    
    def save_file_as(self):
        """Save the current content to a new file.
        
        Returns:
            bool: True if saved successfully
        """
        # Get save path from dialog
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save File As", "", "All Files (*);;Python Files (*.py);;JavaScript Files (*.js)"
        )
        
        if file_path:
            return self.save_file(file_path)
        else:
            return False
    
    def open_file_dialog(self):
        """Open file dialog and load selected file."""
        # Check for unsaved changes
        if self.editor.isModified():
            response = QMessageBox.question(
                self, "Unsaved Changes",
                "The current file has unsaved changes. Do you want to save them?",
                QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel
            )
            
            if response == QMessageBox.StandardButton.Save:
                if not self.save_file():
                    return  # Cancel if save failed
            elif response == QMessageBox.StandardButton.Cancel:
                return
        
        # Show file dialog
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open File", "", 
            "All Files (*);;Python Files (*.py);;JavaScript Files (*.js);;HTML Files (*.html);;CSS Files (*.css)"
        )
        
        if file_path:
            self.load_file(file_path)
    
    def find_text(self):
        """Open find dialog (placeholder)."""
        # In a full implementation, this would open a find dialog
        # For now, just show a message
        QMessageBox.information(self, "Find", "Find functionality would appear here.")
    
    def replace_text(self):
        """Open replace dialog (placeholder)."""
        # In a full implementation, this would open a replace dialog
        # For now, just show a message
        QMessageBox.information(self, "Replace", "Replace functionality would appear here.")
    
    def on_content_changed(self):
        """Handle content changes in the editor."""
        self.content_changed.emit()
    
    def set_text(self, text):
        """Set the editor content.
        
        Args:
            text: Text content to set
        """
        self.editor.setText(text)
    
    def get_text(self):
        """Get the editor content.
        
        Returns:
            str: Current editor content
        """
        return self.editor.text()
    
    def contextMenuEvent(self, event):
        """Custom context menu event.
        
        Args:
            event: Context menu event
        """
        # Create context menu
        menu = QMenu(self)
        
        # Add actions
        cut_action = menu.addAction("Cut")
        copy_action = menu.addAction("Copy")
        paste_action = menu.addAction("Paste")
        menu.addSeparator()
        select_all_action = menu.addAction("Select All")
        menu.addSeparator()
        find_action = menu.addAction("Find...")
        replace_action = menu.addAction("Replace...")
        
        # Connect actions
        cut_action.triggered.connect(self.editor.cut)
        copy_action.triggered.connect(self.editor.copy)
        paste_action.triggered.connect(self.editor.paste)
        select_all_action.triggered.connect(self.editor.selectAll)
        find_action.triggered.connect(self.find_text)
        replace_action.triggered.connect(self.replace_text)
        
        # Show menu
        menu.exec(event.globalPos())


if __name__ == "__main__":
    # Test code
    import sys
    from PyQt6.QtWidgets import QApplication
    from app.theme import apply_theme
    
    app = QApplication(sys.argv)
    apply_theme(app)
    
    editor = CodeEditor()
    editor.resize(800, 600)
    editor.set_text("# Python Code Example\n\ndef example_function():\n    print('Hello World!')\n\n# Call the function\nexample_function()")
    editor.show()
    
    sys.exit(app.exec())