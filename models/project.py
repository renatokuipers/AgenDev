"""
models/project.py
Defines Project data model and handles project-related data validation and operations.
"""
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
from uuid import uuid4, UUID

from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict

# Import file operations utility functions
from utils.file_operations import (
    load_project_metadata, save_project_metadata,
    list_project_files, get_recent_projects
)


class FileInfo(BaseModel):
    """Model representing information about a file in a project."""
    
    model_config = ConfigDict(populate_by_name=True)
    
    name: str
    path: str
    relative_path: str
    extension: str = ""
    size: int = 0
    is_binary: bool = False
    last_modified: datetime = Field(default_factory=datetime.now)
    
    @field_validator('relative_path')
    @classmethod
    def normalize_path(cls, value: str) -> str:
        """Normalize path separators."""
        return value.replace('\\', '/')


class Project(BaseModel):
    """Model representing a project."""
    
    model_config = ConfigDict(populate_by_name=True)
    
    id: UUID = Field(default_factory=uuid4)
    name: str
    path: str
    description: str = ""
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    files: List[FileInfo] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @field_validator('path')
    @classmethod
    def validate_path(cls, value: str) -> str:
        """Validate and normalize the project path."""
        path = Path(value).resolve()
        # Check if the path exists and is a directory
        if not path.exists():
            raise ValueError(f"Project path does not exist: {path}")
        if not path.is_dir():
            raise ValueError(f"Project path is not a directory: {path}")
        return str(path)
    
    @model_validator(mode='after')
    def update_timestamps(self) -> 'Project':
        """Update the updated_at timestamp."""
        self.updated_at = datetime.now()
        return self
    
    def refresh_files(self) -> None:
        """Refresh the list of project files."""
        file_list = list_project_files(self.path)
        self.files = [
            FileInfo(
                name=file["name"],
                path=file["path"],
                relative_path=file["relative_path"],
                extension=file["extension"]
            )
            for file in file_list
        ]
        self.updated_at = datetime.now()
    
    def save_metadata(self) -> bool:
        """Save project metadata to disk."""
        metadata = {
            "id": str(self.id),
            "name": self.name,
            "description": self.description,
            "created_at": self.created_at.isoformat(),
            "updated_at": datetime.now().isoformat(),
            "metadata": self.metadata
        }
        return save_project_metadata(self.path, metadata)
    
    @classmethod
    def create(cls, name: str, path: str, description: str = "") -> 'Project':
        """
        Create a new project.
        
        Args:
            name: Project name
            path: Project path
            description: Project description
            
        Returns:
            Project: Newly created project
        """
        # Create the project instance
        project = cls(
            name=name,
            path=path,
            description=description
        )
        
        # Refresh files and save metadata
        project.refresh_files()
        project.save_metadata()
        
        return project
    
    @classmethod
    def load(cls, path: str) -> 'Project':
        """
        Load a project from a directory.
        
        Args:
            path: Path to the project directory
            
        Returns:
            Project: Loaded project
        """
        # Load metadata from the project directory
        metadata = load_project_metadata(path)
        
        # Create the project instance
        project = cls(
            id=UUID(metadata.get("id", str(uuid4()))),
            name=metadata.get("name", Path(path).name),
            path=path,
            description=metadata.get("description", ""),
            created_at=datetime.fromisoformat(metadata.get("created_at", datetime.now().isoformat())),
            updated_at=datetime.fromisoformat(metadata.get("updated_at", datetime.now().isoformat())),
            metadata=metadata.get("metadata", {})
        )
        
        # Refresh files
        project.refresh_files()
        
        return project
    
    def get_file_by_path(self, relative_path: str) -> Optional[FileInfo]:
        """
        Get a file by its relative path.
        
        Args:
            relative_path: Relative path to the file
            
        Returns:
            Optional[FileInfo]: File info if found, None otherwise
        """
        normalized_path = relative_path.replace('\\', '/')
        for file in self.files:
            if file.relative_path == normalized_path:
                return file
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the project to a dictionary.
        
        Returns:
            Dict[str, Any]: Dictionary representation of the project
        """
        return self.model_dump()


class ProjectManager:
    """Singleton class to manage projects."""
    
    _instance = None
    _current_project: Optional[Project] = None
    _recent_projects: List[Dict[str, Any]] = []
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ProjectManager, cls).__new__(cls)
            cls._recent_projects = get_recent_projects()
        return cls._instance
    
    @property
    def current_project(self) -> Optional[Project]:
        """Get the current active project."""
        return self._current_project
    
    @property
    def recent_projects(self) -> List[Dict[str, Any]]:
        """Get the list of recent projects."""
        return self._recent_projects
    
    def refresh_recent_projects(self) -> None:
        """Refresh the list of recent projects."""
        self._recent_projects = get_recent_projects()
    
    def create_project(self, name: str, path: str, description: str = "") -> Project:
        """
        Create a new project and set it as the current project.
        
        Args:
            name: Project name
            path: Project path
            description: Project description
            
        Returns:
            Project: The newly created project
        """
        project = Project.create(name, path, description)
        self._current_project = project
        return project
    
    def load_project(self, path: str) -> Project:
        """
        Load a project and set it as the current project.
        
        Args:
            path: Path to the project directory
            
        Returns:
            Project: The loaded project
        """
        project = Project.load(path)
        self._current_project = project
        return project
    
    def close_current_project(self) -> None:
        """Close the current project."""
        if self._current_project:
            self._current_project.save_metadata()
            self._current_project = None