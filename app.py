import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

# Import the UI creator
from ui.app import create_gradio_app
from config.settings import settings

# Create the app
demo = create_gradio_app()

if __name__ == "__main__":
    # HuggingFace expects the app to run on host 0.0.0.0
    # and port 7860 by default.
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_api=False
    )
