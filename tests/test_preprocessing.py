import unittest
import numpy as np
from preprocessing.load_data import Trip
from preprocessing.create_features import candidate_features, FEATURE_COLUMNS, segment_distance
from preprocessing.create_labels import reference_alignment, attach_labels

class TestPreprocessing(unittest.TestCase):
    def setUp(self):
        self.trip=Trip('00000001',np.array([[8.0002,49,1],[8.0018,49,2]]),np.array([[8.,49],[8.001,49],[8.002,49]]),np.array([[0,1],[1,2]]))

    def test_segment(self):
        d,f=segment_distance(np.array([1.,1.]),np.array([[0.,0.]]),np.array([[2.,0.]]))
        np.testing.assert_allclose(d,[1]); np.testing.assert_allclose(f,[.5])

    def test_candidates_and_labels(self):
        f=candidate_features(self.trip,k=2,shortlist=2)
        self.assertEqual(len(f),4)
        self.assertTrue(np.isfinite(f[FEATURE_COLUMNS]).all().all())
        a=reference_alignment(self.trip,np.array([0,1]))
        self.assertEqual(a.reference_arc_id.tolist(),[0,1])
        y=attach_labels(f,a)
        self.assertEqual(y.groupby('observation_id').label.sum().tolist(),[1,1])
        self.assertNotIn('label',f.columns)

if __name__=='__main__': unittest.main()
