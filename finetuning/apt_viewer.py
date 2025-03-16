import sys
import subprocess
import os
from PyQt6.QtWidgets import (QApplication, QMainWindow, QTextEdit, QVBoxLayout, 
                            QWidget, QLabel, QPushButton)
from PyQt6.QtCore import QThread, pyqtSignal, Qt
from PyQt6.QtGui import QFont, QTextCursor

class OutputWorker(QThread):
    output_received = pyqtSignal(str)
    error_received = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self.running = True
        
    def run(self):
        # Get the absolute path to the project root directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)  # Just one level up to APT directory
        
        # Debug information
        self.output_received.emit(f"Current directory: {current_dir}")
        self.output_received.emit(f"Project root: {project_root}")
        
        # Try absolute path first
        ml_env_python = "/Users/zhouchaoran/Desktop/APT/my_env/bin/python"
        script_path = os.path.join(current_dir, 'sample.py')
        
        # Debug information
        self.output_received.emit(f"Looking for Python interpreter at: {ml_env_python}")
        self.output_received.emit(f"Script path: {script_path}")
        
        # Verify the paths exist
        if not os.path.exists(ml_env_python):
            self.error_received.emit(f"Python interpreter not found at: {ml_env_python}")
            return
            
        if not os.path.exists(script_path):
            self.error_received.emit(f"Script not found at: {script_path}")
            return
        
        self.output_received.emit(f"Starting script with Python: {ml_env_python}")
        self.output_received.emit(f"Script path: {script_path}")
        
        try:
            # Set the working directory to the script directory
            process = subprocess.Popen(
                [ml_env_python, "-u", script_path],  # Add -u flag to disable buffering
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True,
                cwd=current_dir,  # Set working directory to script location
                env={
                    **os.environ,  # Include current environment variables
                    'PYTHONPATH': project_root,  # Add project root to Python path
                    'PYTHONUNBUFFERED': '1'  # Disable Python buffering
                }
            )
            
            self.output_received.emit("Process started successfully")
            
            import select
            # Set up polling for both stdout and stderr
            poller = select.poll()
            poller.register(process.stdout, select.POLLIN)
            poller.register(process.stderr, select.POLLIN)
            
            while self.running and process.poll() is None:
                # Check for output with a timeout
                events = poller.poll(1000)  # 1 second timeout
                
                for fd, event in events:
                    if fd == process.stdout.fileno():
                        output = process.stdout.readline()
                        if output:
                            self.output_received.emit(output.strip())
                    elif fd == process.stderr.fileno():
                        error = process.stderr.readline()
                        if error:
                            self.error_received.emit(error.strip())
                
                # Periodically check if process is still alive
                if process.poll() is not None:
                    remaining_output, remaining_error = process.communicate()
                    if remaining_output:
                        self.output_received.emit(remaining_output.strip())
                    if remaining_error:
                        self.error_received.emit(remaining_error.strip())
                    break
                    
        except Exception as e:
            self.error_received.emit(f"Error: {str(e)}")
        finally:
            if process.poll() is None:
                try:
                    process.terminate()
                    process.wait(timeout=2)
                except Exception as e:
                    self.error_received.emit(f"Cleanup error: {str(e)}")
    
    def stop(self):
        self.running = False

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("APT Terminal Viewer")
        self.setGeometry(100, 100, 800, 600)
        self.first_output_received = False
        
        # Set window style with white background
        self.setStyleSheet("""
            QMainWindow {
                background-color: #ffffff;
            }
            QTextEdit {
                background-color: #ffffff;
                color: #000000;
                border: none;
                selection-background-color: #000000;
                selection-color: #ffffff;
                font-family: 'Menlo';
                font-size: 18px;
            }
            QLabel {
                color: #000000;
                font-family: 'Menlo';
                font-size: 12px;
            }
            QPushButton {
                background-color: #ffffff;
                color: #000000;
                border: 1px solid #000000;
                padding: 5px;
                font-family: 'Menlo';
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #000000;
                color: #ffffff;
            }
        """)
        
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(10)
        
        # Status label with terminal-style prompt
        self.status_label = QLabel("[ APT Terminal ] Status: Initializing...")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.status_label)
        
        # Create text display
        self.text_display = QTextEdit()
        self.text_display.setReadOnly(True)
        self.text_display.setFont(QFont("Menlo", 12))
        self.text_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.text_display)
        
        # Clear button with terminal style
        self.clear_button = QPushButton("[ Clear Output ]")
        self.clear_button.clicked.connect(self.clear_output)
        self.clear_button.setFixedHeight(30)
        layout.addWidget(self.clear_button)
        
        # Start the worker thread
        self.worker = OutputWorker()
        self.worker.output_received.connect(self.update_display)
        self.worker.error_received.connect(self.show_error)
        self.worker.start()
        
        self.status_label.setText("[ APT Terminal ] Status: Running")
        
    def update_display(self, text):
        try:
            # Skip torch warning
            if "FutureWarning" in text or "torch.load" in text:
                return
                
            if "------------------------------------------------------------" in text:
                # Clear all previous output when first model output is received
                if not self.first_output_received:
                    self.text_display.clear()
                    self.first_output_received = True
                self.text_display.append("\n" + "="*50 + "\n")
            else:
                # Center the text and add it
                if "parameters" not in text and "No meta.pkl" not in text and "Generating" not in text:
                    # This is model output, center it
                    self.text_display.append(text.center(80))
                else:
                    # This is status message, only show if we haven't received first output
                    if not self.first_output_received:
                        self.text_display.append(text)
            
            # Ensure cursor is at the end
            cursor = self.text_display.textCursor()
            cursor.movePosition(QTextCursor.MoveOperation.End)
            self.text_display.setTextCursor(cursor)
            self.text_display.ensureCursorVisible()
        except Exception as e:
            self.show_error(f"Display update error: {str(e)}")
        
    def show_error(self, error):
        # Skip torch warning
        if "FutureWarning" in error or "torch.load" in error:
            return
            
        try:
            if not self.first_output_received:
                self.text_display.append(f"\n[ ERROR ] {error}\n")
                self.status_label.setText("[ APT Terminal ] Status: Error occurred")
                self.status_label.setStyleSheet("color: #ff0000")  # Red color for errors
        except Exception as e:
            print(f"Error in show_error: {str(e)}")
        
    def clear_output(self):
        self.text_display.clear()
        self.first_output_received = False
        
    def closeEvent(self, event):
        try:
            self.status_label.setText("[ APT Terminal ] Status: Shutting down...")
            self.worker.stop()
            self.worker.wait(2000)
            event.accept()
        except Exception as e:
            print(f"Error during shutdown: {str(e)}")
            event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    window = MainWindow()
    window.show()
    
    try:
        sys.exit(app.exec())
    except KeyboardInterrupt:
        print("\nReceived keyboard interrupt, shutting down...")
        window.close() 