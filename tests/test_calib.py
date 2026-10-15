"""Acceptance tests for calib.py (written by Claude, read-only for the intern).

A sample is (dist, true_option): dist maps option -> probability (sums to 1), true_option is the right answer.
"""
import math
import unittest

import calib as c


def over_confident(n=100, acc=0.6):
    # always 0.99 on the pick, right acc*n times
    out = []
    for i in range(n):
        right = i < int(acc * n)
        out.append(({"a": 0.99, "b": 0.01}, "a" if right else "b"))
    return out


def under_confident(n=100):
    # 0.55 on the pick, always right
    return [({"a": 0.55, "b": 0.45}, "a") for _ in range(n)]


class Temp(unittest.TestCase):
    def test_apply_temp(self):
        d = {"a": 0.8, "b": 0.2}
        self.assertEqual(c.apply_temp(d, 1.0), d)
        hot = c.apply_temp(d, 2.0)                      # p ** (1/T), renormalised
        self.assertAlmostEqual(hot["a"], math.sqrt(0.8) / (math.sqrt(0.8) + math.sqrt(0.2)))
        self.assertAlmostEqual(sum(hot.values()), 1.0)
        cold = c.apply_temp(d, 0.5)
        self.assertGreater(cold["a"], 0.8)
        self.assertEqual(c.apply_temp({"a": 1.0, "b": 0.0}, 3.0), {"a": 1.0, "b": 0.0})
        self.assertEqual(c.apply_temp({}, 2.0), {})
        self.assertEqual(list(c.apply_temp({"b": 0.3, "a": 0.7}, 2.0)), ["b", "a"])   # key order kept

    def test_nll(self):
        s = [({"a": 0.5, "b": 0.5}, "a")]
        self.assertAlmostEqual(c.nll(s, 1.0), math.log(2))
        self.assertTrue(math.isfinite(c.nll([({"a": 1.0, "b": 0.0}, "b")], 1.0)))   # floored, not inf
        self.assertTrue(math.isfinite(c.nll([({"a": 1.0}, "zzz")], 1.0)))           # missing option = 0

    def test_fit_temp_direction(self):
        t_over = c.fit_temp(over_confident())
        self.assertGreater(t_over, 1.5)
        top = c.top(c.apply_temp({"a": 0.99, "b": 0.01}, t_over))[1]
        self.assertAlmostEqual(top, 0.6, delta=0.03)                  # calibrated to the real accuracy
        self.assertLess(c.fit_temp(under_confident()), 0.5)
        self.assertEqual(c.fit_temp([]), 1.0)

    def test_fit_temp_bounds(self):
        t = c.fit_temp(under_confident(), lo=0.2, hi=5.0)
        self.assertGreaterEqual(t, 0.2)
        self.assertLessEqual(t, 5.0)
        self.assertLess(c.nll(under_confident(), c.fit_temp(under_confident())), c.nll(under_confident(), 1.0))


class Scores(unittest.TestCase):
    def test_top(self):
        self.assertEqual(c.top({"a": 0.2, "b": 0.7, "c": 0.1}), ("b", 0.7))
        self.assertEqual(c.top({"a": 0.5, "b": 0.5}), ("a", 0.5))     # tie -> first key
        self.assertEqual(c.top({}), (None, 0.0))

    def test_brier(self):
        self.assertAlmostEqual(c.brier([({"a": 1.0, "b": 0.0}, "a")]), 0.0)
        self.assertAlmostEqual(c.brier([({"a": 1.0, "b": 0.0}, "b")]), 2.0)
        self.assertAlmostEqual(c.brier([({"a": 0.5, "b": 0.5}, "a")]), 0.5)
        self.assertAlmostEqual(c.brier([({"a": 1.0}, "z")]), 2.0)      # true option missing from dist counts as p=0
        s = over_confident()
        self.assertLess(c.brier(s, c.fit_temp(s)), c.brier(s, 1.0))
        self.assertEqual(c.brier([]), 0.0)

    def test_ece(self):
        perfect = [({"a": 0.8, "b": 0.2}, "a")] * 8 + [({"a": 0.8, "b": 0.2}, "b")] * 2
        self.assertAlmostEqual(c.ece(perfect), 0.0)
        self.assertAlmostEqual(c.ece(over_confident()), 0.39, places=6)  # conf 0.99, acc 0.6
        s = over_confident()
        self.assertLess(c.ece(s, c.fit_temp(s)), 0.05)
        self.assertEqual(c.ece([]), 0.0)

    def test_report(self):
        r = c.report(over_confident(), 1.0)
        self.assertEqual(r["n"], 100)
        self.assertAlmostEqual(r["acc"], 0.6)
        self.assertAlmostEqual(r["ece"], 0.39, places=6)
        self.assertAlmostEqual(r["brier"], c.brier(over_confident()))
        self.assertAlmostEqual(r["mean_conf"], 0.99)


class Policy(unittest.TestCase):
    def test_route(self):
        self.assertEqual(c.route(0.95), "auto")
        self.assertEqual(c.route(0.9), "auto")
        self.assertEqual(c.route(0.7), "ask")
        self.assertEqual(c.route(0.6), "ask")
        self.assertEqual(c.route(0.59), "escalate")
        self.assertEqual(c.route(None), "escalate")
        self.assertEqual(c.route(0.5, auto=0.8, ask=0.4), "ask")

    def test_auto_threshold(self):
        # conf 0.9 always right, conf 0.7 right half the time
        s = [({"a": 0.9, "b": 0.1}, "a")] * 10 + [({"a": 0.7, "b": 0.3}, "a")] * 5 + [({"a": 0.7, "b": 0.3}, "b")] * 5
        self.assertAlmostEqual(c.auto_threshold(s, 1.0, target_acc=0.95), 0.9)
        self.assertAlmostEqual(c.auto_threshold(s, 1.0, target_acc=0.7), 0.7)   # 15/20 = 0.75 >= 0.7
        self.assertIsNone(c.auto_threshold([({"a": 0.9, "b": 0.1}, "b")], 1.0, target_acc=0.5))
        self.assertIsNone(c.auto_threshold([], 1.0))


class SeqDist(unittest.TestCase):
    def test_mean_and_sum(self):
        lp = {"go left": [-0.1, -0.1], "go right now": [-0.1, -0.1, -0.1]}
        m = c.seq_dist(lp)                                   # default norm="mean": equal mean -> equal p
        self.assertAlmostEqual(m["go left"], 0.5)
        s = c.seq_dist(lp, norm="sum")                       # sum: the longer one is less likely
        self.assertAlmostEqual(s["go left"], math.exp(-0.2) / (math.exp(-0.2) + math.exp(-0.3)))
        self.assertAlmostEqual(sum(s.values()), 1.0)
        self.assertEqual(list(s), ["go left", "go right now"])

    def test_edge(self):
        d = c.seq_dist({"a": [-1.0], "b": []})               # no tokens -> probability 0
        self.assertEqual(d["b"], 0.0)
        self.assertAlmostEqual(d["a"], 1.0)
        self.assertEqual(c.seq_dist({}), {})
        big = c.seq_dist({"a": [-1000.0], "b": [-1001.0]}, norm="sum")   # no underflow
        self.assertAlmostEqual(big["a"], 1 / (1 + math.exp(-1)))
        with self.assertRaises(ValueError):
            c.seq_dist({"a": [-1.0]}, norm="max")


class Save(unittest.TestCase):
    def test_roundtrip(self):
        import os, tempfile
        p = os.path.join(tempfile.mkdtemp(), "cal.json")
        c.save(p, {"T": 1.7, "auto": 0.9, "ask": 0.6})
        self.assertEqual(c.load(p), {"T": 1.7, "auto": 0.9, "ask": 0.6})
        self.assertEqual(c.load(p + ".missing"), {"T": 1.0, "auto": 0.9, "ask": 0.6})   # defaults


if __name__ == "__main__":
    unittest.main()
