"""Student management module for the educational document chat system."""

import json
import re
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import uuid

from src.config import settings


@dataclass
class Student:
    """Student model for the educational system."""
    id: str
    name: str
    created_at: str
    metadata: Dict[str, str]


class StudentManager:
    """Manages student records and operations."""
    
    def __init__(self):
        """Initialize the student manager."""
        self.registry_path = settings.vector_store_path.parent / "students" / "registry.json"
        self._ensure_registry_exists()
    
    def _ensure_registry_exists(self) -> None:
        """Ensure the registry file and directory exist."""
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.registry_path.exists():
            self._save_registry({})
    
    def _load_registry(self) -> Dict[str, Dict]:
        """Load the student registry from disk."""
        with open(self.registry_path, 'r') as f:
            return json.load(f)
    
    def _save_registry(self, registry: Dict[str, Dict]) -> None:
        """Save the student registry to disk."""
        with open(self.registry_path, 'w') as f:
            json.dump(registry, f, indent=2)
    
    def _generate_student_id(self, name: str) -> str:
        """Generate a unique student ID from name."""
        # Clean the name for use in ID
        clean_name = re.sub(r'[^a-zA-Z0-9\s]', '', name.lower())
        clean_name = re.sub(r'\s+', '_', clean_name.strip())
        
        # Add a short UUID to ensure uniqueness
        unique_suffix = str(uuid.uuid4())[:8]
        
        return f"student_{unique_suffix}_{clean_name}"
    
    def create_student(self, name: str, metadata: Optional[Dict[str, str]] = None) -> Student:
        """Create a new student record."""
        if not name or not name.strip():
            raise ValueError("Student name cannot be empty")
        
        student_id = self._generate_student_id(name)
        
        # Check if ID already exists (unlikely but possible)
        registry = self._load_registry()
        if student_id in registry:
            raise ValueError(f"Student ID {student_id} already exists")
        
        # Create student record
        student = Student(
            id=student_id,
            name=name.strip(),
            created_at=datetime.now().isoformat(),
            metadata=metadata or {}
        )
        
        # Save to registry
        registry[student_id] = asdict(student)
        self._save_registry(registry)
        
        # Create student directories
        self._create_student_directories(student_id)
        
        return student
    
    def _create_student_directories(self, student_id: str) -> None:
        """Create necessary directories for a student."""
        base_path = settings.vector_store_path.parent / "students" / student_id
        (base_path / "vector_store").mkdir(parents=True, exist_ok=True)
        (base_path / "graph_state").mkdir(parents=True, exist_ok=True)
    
    def get_student(self, student_id: str) -> Optional[Student]:
        """Get a student by ID."""
        registry = self._load_registry()
        if student_id not in registry:
            return None
        
        student_data = registry[student_id]
        return Student(**student_data)
    
    def list_students(self) -> List[Student]:
        """List all students."""
        registry = self._load_registry()
        students = []
        
        for student_data in registry.values():
            students.append(Student(**student_data))
        
        # Sort by creation date
        students.sort(key=lambda s: s.created_at)
        return students
    
    def delete_student(self, student_id: str) -> bool:
        """Delete a student and all associated data."""
        registry = self._load_registry()
        
        if student_id not in registry:
            return False
        
        # Remove from registry
        del registry[student_id]
        self._save_registry(registry)
        
        # Remove student directories
        import shutil
        student_path = settings.vector_store_path.parent / "students" / student_id
        if student_path.exists():
            shutil.rmtree(student_path)
        
        return True
    
    def student_exists(self, student_id: str) -> bool:
        """Check if a student exists."""
        registry = self._load_registry()
        return student_id in registry


# Global instance
student_manager = StudentManager()