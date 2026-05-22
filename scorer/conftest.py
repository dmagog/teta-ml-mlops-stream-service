"""Делает пакет `src` импортируемым в тестах независимо от рабочей директории."""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
