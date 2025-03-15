"""
components/terminal.py
Implements terminal component, formats command output, and handles terminal text styling.
"""
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPlainTextEdit, 
    QLineEdit, QPushButton, QScrollBar, QFrame
)
from PyQt6.QtCore import Qt, pyqtSignal, pyqtSlot, QProcess, QTimer
from PyQt6.QtGui import QColor, QTextCharFormat, QTextCursor, QFont, QTextOption


class AnsiEscapeParser:
    """Parser for ANSI escape sequences in terminal output."""
    
    # ANSI color codes
    COLORS = {
        '30': QColor("#000000"),  # Black
        '31': QColor("#CC0000"),  # Red
        '32': QColor("#4E9A06"),  # Green
        '33': QColor("#C4A000"),  # Yellow
        '34': QColor("#3465A4"),  # Blue
        '35': QColor("#75507B"),  # Magenta
        '36': QColor("#06989A"),  # Cyan
        '37': QColor("#D3D7CF"),  # White
        # Bright variants
        '90': QColor("#555753"),  # Bright Black (Gray)
        '91': QColor("#EF2929"),  # Bright Red
        '92': QColor("#8AE234"),  # Bright Green
        '93': QColor("#FCE94F"),  # Bright Yellow
        '94': QColor("#729FCF"),  # Bright Blue
        '95': QColor("#AD7FA8"),  # Bright Magenta
        '96': QColor("#34E2E2"),  # Bright Cyan
        '97': QColor("#EEEEEC"),  # Bright White
    }
    
    @staticmethod
    def parse_text(text, base_format):
        """Parse text with ANSI escape sequences and return formatted segments.
        
        Args:
            text: Text containing ANSI escape sequences
            base_format: Base text format to use
            
        Returns:
            list: List of tuples (text, format)
        """
        result = []
        current_text = ""
        current_format = QTextCharFormat(base_format)
        i = 0
        
        while i < len(text):
            if text[i] == '\033' and i + 1 < len(text) and text[i + 1] == '[':
                # Found escape sequence start
                if current_text:
                    result.append((current_text, QTextCharFormat(current_format)))
                    current_text = ""
                
                # Find the end of the escape sequence
                end = text.find('m', i)
                if end == -1:
                    # No end found, treat as normal text
                    current_text += text[i]
                    i += 1
                    continue
                
                # Parse the escape sequence
                sequence = text[i + 2:end]
                codes = sequence.split(';')
                
                for code in codes:
                    if code == '0':
                        # Reset to default
                        current_format = QTextCharFormat(base_format)
                    elif code == '1':
                        # Bold
                        current_format.setFontWeight(QFont.Weight.Bold)
                    elif code == '3':
                        # Italic
                        current_format.setFontItalic(True)
                    elif code == '4':
                        # Underline
                        current_format.setFontUnderline(True)
                    elif code in AnsiEscapeParser.COLORS:
                        # Foreground color
                        current_format.setForeground(AnsiEscapeParser.COLORS[code])
                    elif code.startswith('4') and code[1:] in AnsiEscapeParser.COLORS:
                        # Background color (40-47, 100-107)
                        current_format.setBackground(AnsiEscapeParser.COLORS[code[1:]])
                
                # Skip the escape sequence
                i = end + 1
            else:
                current_text += text[i]
                i += 1
        
        # Add any remaining text
        if current_text:
            result.append((current_text, current_format))
        
        return result


class TerminalDisplay(QPlainTextEdit):
    """Widget for displaying terminal output with ANSI color support."""
    
    def __init__(self, parent=None):
        """Initialize terminal display widget."""
        super().__init__(parent)
        self.setObjectName("terminalDisplay")
        
        # Configure display properties
        self.setReadOnly(True)
        self.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.setWordWrapMode(QTextOption.WrapMode.NoWrap)
        self.setUndoRedoEnabled(False)
        self.setMaximumBlockCount(5000)  # Limit for performance
        
        # Set font
        font = QFont("Consolas", 10)
        font.setFixedPitch(True)
        self.setFont(font)
        
        # Set colors (base format)
        self.base_format = QTextCharFormat()
        self.base_format.setFont(font)
        self.base_format.setForeground(QColor("#C0C0C0"))  # Default terminal text color
        
        # Set placeholder text
        self.setPlaceholderText("Terminal output will appear here...")
    
    def append_text(self, text):
        """Append text to the terminal display with ANSI color support.
        
        Args:
            text: Text to append, may contain ANSI escape sequences
        """
        # Parse ANSI escape sequences
        segments = AnsiEscapeParser.parse_text(text, self.base_format)
        
        # Get cursor for editing
        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        
        # Add each segment with its format
        for text_segment, format_segment in segments:
            cursor.insertText(text_segment, format_segment)
        
        # Scroll to the new content
        self.setTextCursor(cursor)
        self.ensureCursorVisible()
    
    def clear_terminal(self):
        """Clear the terminal display."""
        self.clear()


class CommandInput(QLineEdit):
    """Input field for terminal commands."""
    
    command_entered = pyqtSignal(str)
    
    def __init__(self, parent=None):
        """Initialize command input field."""
        super().__init__(parent)
        self.setObjectName("commandInput")
        self.setPlaceholderText("Enter command...")
        
        # Command history
        self.command_history = []
        self.history_index = -1
        
        # Connect signals
        self.returnPressed.connect(self.on_return_pressed)
    
    def on_return_pressed(self):
        """Handle return key press."""
        command = self.text()
        if command:
            # Add to history
            self.command_history.append(command)
            self.history_index = len(self.command_history)
            
            # Emit signal
            self.command_entered.emit(command)
            
            # Clear input
            self.clear()
    
    def keyPressEvent(self, event):
        """Handle key press events for command history navigation.
        
        Args:
            event: Key event
        """
        if event.key() == Qt.Key.Key_Up:
            # Navigate up in history
            if self.command_history and self.history_index > 0:
                self.history_index -= 1
                self.setText(self.command_history[self.history_index])
                self.selectAll()
        elif event.key() == Qt.Key.Key_Down:
            # Navigate down in history
            if self.command_history and self.history_index < len(self.command_history) - 1:
                self.history_index += 1
                self.setText(self.command_history[self.history_index])
                self.selectAll()
            elif self.history_index == len(self.command_history) - 1:
                # At the end of history, clear input
                self.history_index = len(self.command_history)
                self.clear()
        else:
            # Default handling
            super().keyPressEvent(event)


class Terminal(QWidget):
    """Terminal component for command execution and output display."""
    
    def __init__(self, parent=None):
        """Initialize terminal component."""
        super().__init__(parent)
        self.setObjectName("terminal")
        
        # Process for command execution
        self.process = QProcess(self)
        
        self.setup_ui()
        self.setup_connections()
        
        # Start with a welcome message
        self.display.append_text("Terminal Ready\n$ ")
    
    def setup_ui(self):
        """Set up terminal UI components."""
        # Main layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Terminal display
        self.display = TerminalDisplay(self)
        
        # Command input container
        cmd_container = QWidget()
        cmd_layout = QHBoxLayout(cmd_container)
        cmd_layout.setContentsMargins(4, 4, 4, 4)
        cmd_layout.setSpacing(4)
        
        # Command prompt
        self.prompt_label = QLabel("$")
        self.prompt_label.setFixedWidth(15)
        self.prompt_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        
        # Command input
        self.command_input = CommandInput()
        
        # Clear button
        self.clear_button = QPushButton("Clear")
        self.clear_button.setObjectName("clearButton")
        self.clear_button.setToolTip("Clear terminal")
        
        # Add widgets to command layout
        cmd_layout.addWidget(self.prompt_label)
        cmd_layout.addWidget(self.command_input)
        cmd_layout.addWidget(self.clear_button)
        
        # Add widgets to main layout
        layout.addWidget(self.display, 1)
        layout.addWidget(cmd_container, 0)
    
    def setup_connections(self):
        """Set up signal/slot connections."""
        # Command input signals
        self.command_input.command_entered.connect(self.execute_command)
        
        # Clear button
        self.clear_button.clicked.connect(self.display.clear_terminal)
        
        # Process signals
        self.process.readyReadStandardOutput.connect(self.handle_stdout)
        self.process.readyReadStandardError.connect(self.handle_stderr)
        self.process.finished.connect(self.handle_finished)
    
    def execute_command(self, command):
        """Execute a command in the terminal.
        
        Args:
            command: Command to execute
        """
        # Display the command
        self.display.append_text(f"{command}\n")
        
        # On Windows, use cmd.exe to execute commands
        if self.process.state() == QProcess.ProcessState.NotRunning:
            self.process.setProgram("cmd.exe")
            self.process.setArguments(["/c", command])
            self.process.start()
    
    def handle_stdout(self):
        """Handle standard output from the process."""
        data = self.process.readAllStandardOutput()
        text = bytes(data).decode('utf-8', errors='replace')
        # Replace Windows line endings with Unix style
        text = text.replace('\r\n', '\n')
        self.display.append_text(text)
    
    def handle_stderr(self):
        """Handle standard error from the process."""
        data = self.process.readAllStandardError()
        text = bytes(data).decode('utf-8', errors='replace')
        # Replace Windows line endings with Unix style
        text = text.replace('\r\n', '\n')
        # Use ANSI escape sequence for red text
        self.display.append_text(f"\033[31m{text}\033[0m")
    
    def handle_finished(self, exit_code, exit_status):
        """Handle process completion.
        
        Args:
            exit_code: Process exit code
            exit_status: Process exit status
        """
        # Append prompt for next command
        self.display.append_text("\n$ ")
        
        # Set focus to command input
        self.command_input.setFocus()
    
    def simulate_command(self, command, output):
        """Simulate a command execution with predefined output (for demo purposes).
        
        Args:
            command: Command to simulate
            output: Simulated output
        """
        # Display the command
        self.display.append_text(f"{command}\n")
        
        # Simulate delay
        QTimer.singleShot(300, lambda: self.display.append_text(output + "\n$ "))


if __name__ == "__main__":
    # Test code
    import sys
    from PyQt6.QtWidgets import QApplication, QLabel
    from app.theme import apply_theme
    
    app = QApplication(sys.argv)
    apply_theme(app)
    
    term = Terminal()
    term.resize(800, 400)
    term.show()
    
    # Simulate some commands for demonstration
    QTimer.singleShot(500, lambda: term.simulate_command(
        "npm install",
        "added 1250 packages, and audited 1251 packages in 5s\n\n" +
        "125 packages are looking for funding\n" +
        "  run `npm fund` for details\n\n" +
        "\033[32mfound 0 vulnerabilities\033[0m"
    ))
    
    sys.exit(app.exec())