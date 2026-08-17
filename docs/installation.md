# Installation & Environment Setup Guide

## Prerequisites

- **Python Version**: Python 3.8+ (Tested on Python 3.13)
- **Operating System**: Windows / Linux / macOS
- **Hardware**: CPU supported; NVIDIA GPU with CUDA recommended for high FPS

## Setup Steps

1. **Clone Repository & Navigate to Folder**:
   ```bash
   git clone <repository_url>
   cd "AI-Based Vehicle Classification for Urban Air and Noise Pollution Monitoring"
   ```

2. **Create Virtual Environment**:
   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On Linux/macOS:
   source venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify Installation**:
   ```bash
   python run.py --test
   ```
