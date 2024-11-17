import os
import subprocess
import pytest

def get_python_files():
    """Recursively get all Python files in the project."""
    python_files = []
    for root, _, files in os.walk('.'):
        if 'venv' in root or 'ENV' in root or '.git' in root:
            continue
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    return python_files

def test_flake8():
    """Test if all Python files pass flake8."""
    python_files = get_python_files()
    
    # Run flake8 with specific rules
    cmd = ['flake8', '--max-line-length=100', '--ignore=E402,W503']
    cmd.extend(python_files)
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        pytest.fail(f"Flake8 found issues:\n{result.stdout}\n{result.stderr}")

def test_black():
    """Test if all Python files are formatted according to black."""
    python_files = get_python_files()
    
    # Check black formatting
    cmd = ['black', '--check', '--line-length=100']
    cmd.extend(python_files)
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        pytest.fail(f"Black formatting issues found:\n{result.stdout}\n{result.stderr}")

def test_isort():
    """Test if imports are properly sorted."""
    python_files = get_python_files()
    
    # Check import sorting
    cmd = ['isort', '--check-only', '--profile', 'black']
    cmd.extend(python_files)
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        pytest.fail(f"Import sorting issues found:\n{result.stdout}\n{result.stderr}") 