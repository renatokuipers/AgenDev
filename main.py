"""
main.py
Application entry point that initializes and runs the main window.
"""
import sys
import os
from PyQt6.QtWidgets import QApplication

# Add the project root to the Python path to enable absolute imports
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Import main components
from app.window import MainWindow, run_application
from app.theme import apply_theme


def main():
    """
    Main entry point function for the application.
    
    Sets up the environment, creates the application window,
    and starts the application event loop.
    """
    # Ensure the current directory is the project root
    os.chdir(project_root)
    
    # Create the application
    app = QApplication(sys.argv)
    
    # Apply the dark theme
    apply_theme(app)
    
    # Create and show the main window
    window = MainWindow()
    window.show()
    
    # Start the application event loop
    sys.exit(app.exec())


if __name__ == "__main__":
    """
    Entry point for the application.
    
    This block is executed when the script is run directly
    (not when imported as a module).
    """
    # Check if the run_application function exists, use it if available
    if 'run_application' in globals():
        run_application()
    else:
        # Otherwise use our local main function
        main()