import sys
import os
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from maba.config import Config
from maba.model import Model
from maba.tokenizer import Tokenizer
from maba.generate import spec_gen

class TestSpeculativeGeneration(unittest.TestCase):
    def test_spec_decode(self):
        cfg = Config()
        model = Model(cfg)
        tok = Tokenizer()
        prompt = "def delta_rule"

        txt, acc, steps = spec_gen(
            model=model,
            tokenizer=tok,
            prompt=prompt,
            max_new_tokens=15,
            device="cpu"
        )
        self.assertIsInstance(txt, str)
        self.assertGreater(len(txt), len(prompt))
        self.assertGreater(steps, 0)

if __name__ == "__main__":
    unittest.main()
