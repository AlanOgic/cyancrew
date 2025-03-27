#!/usr/bin/env python3
"""
Setup script for the Product Research and Sales Guide CrewAI Project.
This script automates the creation of a virtual environment and installation of dependencies.
"""

import os
import sys
import subprocess
import platform

def check_python_version():
    """Check if Python version is 3.8 or higher."""
    required_major = 3
    required_minor = 8
    
    if sys.version_info.major < required_major or \
       (sys.version_info.major == required_major and sys.version_info.minor < required_minor):
        print(f"Error: Python {required_major}.{required_minor}+ is required.")
        print(f"Current Python version: {sys.version}")
        sys.exit(1)
    
    print(f"Python version check passed: {sys.version}")

def create_virtual_environment():
    """Create a virtual environment."""
    if os.path.exists("venv"):
        print("Virtual environment already exists.")
        return
    
    print("Creating virtual environment...")
    try:
        subprocess.run([sys.executable, "-m", "venv", "venv"], check=True)
        print("Virtual environment created successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error creating virtual environment: {e}")
        sys.exit(1)

def get_activation_command():
    """Get the command to activate the virtual environment based on the OS."""
    if platform.system() == "Windows":
        return os.path.join("venv", "Scripts", "activate")
    else:  # macOS or Linux
        return f"source {os.path.join('venv', 'bin', 'activate')}"

def install_dependencies():
    """Install dependencies from requirements.txt."""
    print("Installing dependencies...")
    
    # Determine the pip executable path based on the OS
    if platform.system() == "Windows":
        pip_path = os.path.join("venv", "Scripts", "pip")
    else:  # macOS or Linux
        pip_path = os.path.join("venv", "bin", "pip")
    
    try:
        subprocess.run([pip_path, "install", "-r", "requirements.txt"], check=True)
        print("Dependencies installed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error installing dependencies: {e}")
        sys.exit(1)

def main():
    """Main function to set up the project."""
    print("\n" + "="*50)
    print("Product Research and Sales Guide CrewAI Project Setup")
    print("="*50 + "\n")
    
    # Check Python version
    check_python_version()
    
    # Create virtual environment
    create_virtual_environment()
    
    # Install dependencies
    install_dependencies()
    
    # Print instructions
    activation_cmd = get_activation_command()
    
    print("\n" + "="*50)
    print("Setup completed successfully!")
    print("="*50 + "\n")
    
    print("To use the project:")
    print(f"1. Activate the virtual environment:")
    print(f"   {activation_cmd}")
    print("2. Run the script:")
    print("   python crew-1.py")
    print("3. When finished, deactivate the virtual environment:")
    print("   deactivate\n")

if __name__ == "__main__":
    main()
