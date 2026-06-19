"""
File Manager Module

Responsible for:
- Creating directories
- Writing files to disk
- Tracking created files
- Handling file permission errors
"""

import os
from typing import List, Tuple


def create_or_update_file(file_path: str, file_content: str) -> Tuple[bool, str, List[str]]:
    """
    Creates or updates a file with the given content.
    
    Automatically creates parent directories if they don't exist.
    
    Args:
        file_path: Relative or absolute path to the file (e.g., "src/app.py" or "requirements.txt")
        file_content: The exact content to write to the file
    
    Returns:
        Tuple of (success: bool, error_message: str, files_created: List[str])
        - success: True if file was created/updated successfully
        - error_message: Error details if failed, empty string if succeeded
        - files_created: List of files that were successfully created/modified
    
    Example:
        success, error, files = create_or_update_file("src/app.py", "print('hello')")
        if success:
            print(f"Created: {files}")  # Output: ['src/app.py']
        else:
            print(f"Failed: {error}")
    """
    print(f"\n--- FILE MANAGER: Creating/Updating File ---")
    print(f"File Path: {file_path}")
    print(f"Content Length: {len(file_content)} characters")
    
    try:
        # Step 1: Extract directory path
        dir_path = os.path.dirname(file_path)
        
        # Step 2: Create parent directories if needed
        if dir_path:
            print(f"Creating directory: {dir_path}")
            os.makedirs(dir_path, exist_ok=True)
        
        # Step 3: Write file content
        print(f"Writing file: {file_path}")
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(file_content)
        
        # Step 4: Verify file was created
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
            log_msg = f"Successfully created/updated: {file_path} ({file_size} bytes)"
            print(f"✓ {log_msg}")
            return True, "", [file_path]
        else:
            error_msg = f"File was not created even though write succeeded: {file_path}"
            print(f"✗ {error_msg}")
            return False, error_msg, []
    
    except PermissionError as e:
        error_msg = f"Permission denied while creating {file_path}: {str(e)}"
        print(f"✗ {error_msg}")
        return False, error_msg, []
    
    except OSError as e:
        error_msg = f"OS error while creating {file_path}: {str(e)}"
        print(f"✗ {error_msg}")
        return False, error_msg, []
    
    except Exception as e:
        error_msg = f"Unexpected error while creating {file_path}: {str(e)}"
        print(f"✗ {error_msg}")
        return False, error_msg, []


def verify_file_exists(file_path: str) -> bool:
    """
    Verifies that a file exists at the given path.
    
    Args:
        file_path: Path to check
    
    Returns:
        True if file exists, False otherwise
    """
    return os.path.exists(file_path) and os.path.isfile(file_path)


def list_created_files(directory: str = ".") -> List[str]:
    """
    Lists all files in a directory (useful for verification).
    
    Args:
        directory: Directory to list (default is current directory)
    
    Returns:
        List of file paths
    """
    files = []
    for root, dirs, filenames in os.walk(directory):
        for filename in filenames:
            files.append(os.path.join(root, filename))
    return files


# --- Local Testing Block ---
if __name__ == "__main__":
    import tempfile
    
    print("Testing File Manager...")
    
    # Test 1: Create file in current directory
    print("\n--- Test 1: Simple file ---")
    success, error, files = create_or_update_file("test_file.txt", "Hello World")
    print(f"Success: {success}, Files: {files}")
    if os.path.exists("test_file.txt"):
        os.remove("test_file.txt")
        print("Cleaned up test file")
    
    # Test 2: Create file with nested directories
    print("\n--- Test 2: Nested directories ---")
    success, error, files = create_or_update_file("test_dir/subdir/app.py", "print('hello')")
    print(f"Success: {success}, Files: {files}")
    
    # Test 3: Verify file exists
    print("\n--- Test 3: Verify file ---")
    exists = verify_file_exists("test_dir/subdir/app.py")
    print(f"File exists: {exists}")
    
    # Cleanup
    import shutil
    if os.path.exists("test_dir"):
        shutil.rmtree("test_dir")
        print("Cleaned up test directory")
