import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel

app = QApplication(sys.argv)
window = QMainWindow()
window.setGeometry(100, 100, 300, 200)
window.setWindowTitle('PyQt6 Test')
label = QLabel('PyQt6 is working!', parent=window)
label.setGeometry(50, 80, 200, 30)
window.show()
sys.exit(app.exec()) 