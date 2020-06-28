import sys
import unittest
from PyQt5.QtTest import QTest
from PyQt5.QtGui import QGuiApplication
from PyQt5.QtCore import Qt
import test2_ui
from test2_ui import *

class EmotionTest(unittest.TestCase):
    def test(self):
        self.testapp = Ui_Form()
        text = self.testapp.control_bt.__getattribute__
        self.assertEqual(text, "Start")

if __name__=='__main__':
    T = EmotionTest()
    T.test()
