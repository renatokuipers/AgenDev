"""
utils/file_operations.py
Provides utility functions for file operations, project management, and file system interaction.
"""
import os
import json
import shutil
from datetime import datetime
from typing import List, Dict, Any, Optional, Union
from pathlib import Path

# Default locations
DEFAULT_PROJECTS_DIR = os.path.expanduser("~/same/projects")
DEFAULT_CONFIG_DIR = os.path.expanduser("~/same/config")


def ensure_directory_exists(directory_path: str) -> None:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        directory_path: Path to the directory
    """
    os.makedirs(directory_path, exist_ok=True)


def get_recent_projects(limit: int = 10) -> List[Dict[str, Any]]:
    """
    Get a list of recent projects.
    
    Args:
        limit: Maximum number of projects to return
        
    Returns:
        List of project dictionaries with name and path
    """
    # Ensure config directory exists
    ensure_directory_exists(DEFAULT_CONFIG_DIR)
    
    # Path to recent projects file
    recent_file = os.path.join(DEFAULT_CONFIG_DIR, "recent_projects.json")
    
    # If the file doesn't exist, return an empty list
    if not os.path.exists(recent_file):
        return []
    
    try:
        # Read and parse the recent projects file
        with open(recent_file, "r", encoding="utf-8") as f:
            projects = json.load(f)
        
        # Sort by last accessed time (most recent first) and limit
        projects.sort(key=lambda p: p.get("last_accessed", 0), reverse=True)
        return projects[:limit]
    except Exception as e:
        print(f"Error loading recent projects: {e}")
        return []


def add_recent_project(name: str, path: str) -> None:
    """
    Add a project to the recent projects list.
    
    Args:
        name: Project name
        path: Project path
    """
    # Ensure config directory exists
    ensure_directory_exists(DEFAULT_CONFIG_DIR)
    
    # Path to recent projects file
    recent_file = os.path.join(DEFAULT_CONFIG_DIR, "recent_projects.json")
    
    # Current time as timestamp
    timestamp = int(datetime.now().timestamp())
    
    # New project entry
    project_entry = {
        "name": name,
        "path": path,
        "last_accessed": timestamp
    }
    
    # Load existing projects or create empty list
    if os.path.exists(recent_file):
        try:
            with open(recent_file, "r", encoding="utf-8") as f:
                projects = json.load(f)
        except:
            projects = []
    else:
        projects = []
    
    # Check if project already exists in the list
    for i, project in enumerate(projects):
        if project.get("path") == path:
            # Update existing entry and move to the top
            projects.pop(i)
            projects.insert(0, project_entry)
            break
    else:
        # Add new project to the beginning
        projects.insert(0, project_entry)
    
    # Save updated projects list
    try:
        with open(recent_file, "w", encoding="utf-8") as f:
            json.dump(projects, f, indent=2)
    except Exception as e:
        print(f"Error saving recent projects: {e}")


def create_project_directory(project_name: str) -> str:
    """
    Create a new project directory.
    
    Args:
        project_name: Name of the project
        
    Returns:
        Path to the created project directory
    """
    # Ensure projects directory exists
    ensure_directory_exists(DEFAULT_PROJECTS_DIR)
    
    # Create a valid directory name from the project name
    dir_name = project_name.lower().replace(" ", "_")
    
    # Check if directory already exists and add timestamp if needed
    project_dir = os.path.join(DEFAULT_PROJECTS_DIR, dir_name)
    if os.path.exists(project_dir):
        timestamp = int(datetime.now().timestamp())
        project_dir = os.path.join(DEFAULT_PROJECTS_DIR, f"{dir_name}_{timestamp}")
    
    # Create the project directory
    os.makedirs(project_dir)
    
    # Add to recent projects
    add_recent_project(project_name, project_dir)
    
    return project_dir


def list_directory_contents(directory: str) -> Dict[str, List[str]]:
    """
    List files and subdirectories in a directory.
    
    Args:
        directory: Path to the directory
        
    Returns:
        Dictionary with 'files' and 'directories' lists
    """
    result = {
        "files": [],
        "directories": []
    }
    
    try:
        # Get all items in the directory
        items = os.listdir(directory)
        
        # Separate files and directories
        for item in items:
            item_path = os.path.join(directory, item)
            if os.path.isdir(item_path):
                result["directories"].append(item)
            else:
                result["files"].append(item)
                
        # Sort alphabetically
        result["files"].sort()
        result["directories"].sort()
    except Exception as e:
        print(f"Error listing directory {directory}: {e}")
    
    return result


def read_file_content(file_path: str) -> Optional[str]:
    """
    Read the content of a file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        File content as string, or None if reading fails
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        # Try reading as binary file
        try:
            with open(file_path, "rb") as f:
                return f.read().decode("utf-8", errors="replace")
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            return None
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None


def write_file_content(file_path: str, content: str) -> bool:
    """
    Write content to a file.
    
    Args:
        file_path: Path to the file
        content: Content to write
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Ensure the directory exists
        directory = os.path.dirname(file_path)
        if directory:
            ensure_directory_exists(directory)
            
        # Write the content
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        return True
    except Exception as e:
        print(f"Error writing to file {file_path}: {e}")
        return False


def copy_file(source: str, destination: str) -> bool:
    """
    Copy a file from source to destination.
    
    Args:
        source: Source file path
        destination: Destination file path
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Ensure the destination directory exists
        destination_dir = os.path.dirname(destination)
        if destination_dir:
            ensure_directory_exists(destination_dir)
            
        # Copy the file
        shutil.copy2(source, destination)
        return True
    except Exception as e:
        print(f"Error copying {source} to {destination}: {e}")
        return False


def delete_file(file_path: str) -> bool:
    """
    Delete a file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        os.remove(file_path)
        return True
    except Exception as e:
        print(f"Error deleting file {file_path}: {e}")
        return False


def create_directory(directory_path: str) -> bool:
    """
    Create a directory.
    
    Args:
        directory_path: Path to the directory
        
    Returns:
        True if successful, False otherwise
    """
    try:
        os.makedirs(directory_path, exist_ok=True)
        return True
    except Exception as e:
        print(f"Error creating directory {directory_path}: {e}")
        return False


def delete_directory(directory_path: str, recursive: bool = True) -> bool:
    """
    Delete a directory.
    
    Args:
        directory_path: Path to the directory
        recursive: Whether to recursively delete all contents
        
    Returns:
        True if successful, False otherwise
    """
    try:
        if recursive:
            shutil.rmtree(directory_path)
        else:
            os.rmdir(directory_path)
        return True
    except Exception as e:
        print(f"Error deleting directory {directory_path}: {e}")
        return False


def get_file_extension(file_path: str) -> str:
    """
    Get the extension of a file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        File extension (without the dot)
    """
    return os.path.splitext(file_path)[1][1:].lower()


def is_binary_file(file_path: str) -> bool:
    """
    Check if a file is binary or text.
    
    Args:
        file_path: Path to the file
        
    Returns:
        True if the file is binary, False otherwise
    """
    text_extensions = [
        "txt", "py", "js", "html", "css", "json", "xml", "md", "rst", 
        "csv", "yml", "yaml", "ini", "cfg", "conf", "log", "sh", "bat"
    ]
    
    # Check by extension first
    extension = get_file_extension(file_path)
    if extension in text_extensions:
        return False
    
    # Try to read the file as text
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            f.read(1024)  # Read a small chunk
        return False  # If we got here, it's a text file
    except UnicodeDecodeError:
        return True  # It's a binary file if we can't decode as UTF-8
    except Exception:
        # If we can't read the file, assume it's binary
        return True


def get_file_info(file_path: str) -> Dict[str, Any]:
    """
    Get information about a file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Dictionary with file information
    """
    try:
        stats = os.stat(file_path)
        return {
            "name": os.path.basename(file_path),
            "path": file_path,
            "size": stats.st_size,
            "created": stats.st_ctime,
            "modified": stats.st_mtime,
            "extension": get_file_extension(file_path),
            "is_binary": is_binary_file(file_path)
        }
    except Exception as e:
        print(f"Error getting info for {file_path}: {e}")
        return {
            "name": os.path.basename(file_path),
            "path": file_path,
            "error": str(e)
        }


def get_relative_path(path: str, base_path: str) -> str:
    """
    Get the path relative to a base path.
    
    Args:
        path: Absolute path
        base_path: Base path to make the path relative to
        
    Returns:
        Relative path
    """
    try:
        return os.path.relpath(path, base_path)
    except:
        return path


def get_absolute_path(relative_path: str, base_path: str) -> str:
    """
    Get the absolute path from a path relative to a base path.
    
    Args:
        relative_path: Relative path
        base_path: Base path
        
    Returns:
        Absolute path
    """
    return os.path.normpath(os.path.join(base_path, relative_path))


def list_project_files(project_path: str) -> List[Dict[str, Any]]:
    """
    Recursively list all files in a project directory.
    
    Args:
        project_path: Path to the project directory
        
    Returns:
        List of file information dictionaries
    """
    files = []
    
    try:
        for root, _, filenames in os.walk(project_path):
            for filename in filenames:
                file_path = os.path.join(root, filename)
                rel_path = get_relative_path(file_path, project_path)
                
                # Skip hidden files and directories
                if any(part.startswith('.') for part in rel_path.split(os.sep)):
                    continue
                
                files.append({
                    "name": filename,
                    "path": file_path,
                    "relative_path": rel_path,
                    "extension": get_file_extension(file_path)
                })
    except Exception as e:
        print(f"Error listing project files in {project_path}: {e}")
    
    return files


def save_project_metadata(project_path: str, metadata: Dict[str, Any]) -> bool:
    """
    Save project metadata to a file.
    
    Args:
        project_path: Path to the project directory
        metadata: Project metadata
        
    Returns:
        True if successful, False otherwise
    """
    metadata_file = os.path.join(project_path, ".same_project.json")
    
    try:
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        return True
    except Exception as e:
        print(f"Error saving project metadata: {e}")
        return False


def load_project_metadata(project_path: str) -> Dict[str, Any]:
    """
    Load project metadata from a file.
    
    Args:
        project_path: Path to the project directory
        
    Returns:
        Project metadata dictionary
    """
    metadata_file = os.path.join(project_path, ".same_project.json")
    
    # Default metadata if file doesn't exist
    if not os.path.exists(metadata_file):
        return {
            "name": os.path.basename(project_path),
            "path": project_path,
            "created": datetime.now().isoformat()
        }
    
    try:
        with open(metadata_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading project metadata: {e}")
        return {
            "name": os.path.basename(project_path),
            "path": project_path,
            "error": str(e)
        }