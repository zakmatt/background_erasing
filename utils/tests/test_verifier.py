import numpy as np

from unittest import TestCase
from verifier import (
    Verifier,
    DirectoryNotExisting,
    NoSuchArchitectureImpemented
)

MODEL_PATH = ''
ARCHITECTURE = ''
MODEL_WEIGHTS = ''

class TestVerifier(TestCase):
    """Class testing a batch generator"""

    def setUp(self):
        self.verifier = Verifier(MODEL_PATH, ARCHITECTURE, MODEL_WEIGHTS)
