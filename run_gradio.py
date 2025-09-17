#!/usr/bin/env python3
"""
Simple launcher script for MenuGen Gradio App
"""

if __name__ == "__main__":
    from app_gradio import create_interface
    
    print("🍽️ Starting MenuGen Gradio App...")
    print("🔮 Loading models may take a moment on first run...")
    print("📱 App will be available at: http://localhost:7860")
    print("🌐 Or at the URL shown below if running on a server")
    
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        show_error=True,
        inbrowser=True  # Try to open browser automatically
    )

