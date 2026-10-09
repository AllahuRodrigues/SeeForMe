import json, pathlib, tempfile, unittest
from sample_coverage import sample_catalog
BASE=pathlib.Path(__file__).parent

class CatalogTests(unittest.TestCase):
    def test_no_duplicate_identifiers(self):
        s=json.loads((BASE/'scenario_catalog.json').read_text())['scenarios']
        self.assertEqual(len({x['scenario_id'] for x in s}),len(s))
        self.assertGreaterEqual(len(s),150)
    def test_catalog_does_not_claim_ground_truth(self):
        s=json.loads((BASE/'scenario_catalog.json').read_text())
        self.assertTrue(all(x['scenario_status']=='catalog_only' and x['annotation_status']=='not_collected' for x in s['scenarios']))
    def test_ood_emergency_not_marked_for_training(self):
        rows=sample_catalog(1,42)
        self.assertTrue(all(x['split']=='ood_test' for x in rows if x['scope']=='ood_only'))
        self.assertTrue(all(x['labels']=='NOT_ANNOTATED' for x in rows))
    def test_reproducible_and_distinct(self):
        a=sample_catalog(3,1337);b=sample_catalog(3,1337)
        self.assertEqual(a,b)
        self.assertEqual(len(a),462)
        self.assertEqual(len({r['instance_id'] for r in a}),462)
        self.assertNotEqual(a,sample_catalog(3,8))
    def test_sign_coverage(self):
        s=json.loads((BASE/'sign_taxonomy.json').read_text())['categories']
        self.assertGreaterEqual(sum(map(len,s.values())),80)
        self.assertTrue({'airport','vertical_access','emergency','negative_text'}<=s.keys())
if __name__=='__main__': unittest.main()
