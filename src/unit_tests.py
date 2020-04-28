import unittests
import random
import numpy as np

import data_utils


class TestTemplate(unittest.TestCase):

    def test_that():
        self.assertTrue(True)
        self.assertEqual(1, 1)
        pass


class TestTemplate(unittest.TestCase):

    def test_that():
        from main import get_data

        self.assertTrue()
        self.assertEqual()

        # A test to assert that every data point has an entry for each property
        properties = set(df['Property'].tolist())
        print(f'There are {len(properties)} many properties')
        for k, v in instances.items():
            for k2, v2 in v.items():
                for p in properties:
                    try:
                        assert p in v2
                    except Exception:
                        print(f'Uh oh! Missing {p}')
                        print(k2)
                        print(v2)
                        breakpoint()


    # Little test to make sure arg_indices make sense
    #for split in SPLITS:
    #    for pt in proto_instances[split]:
    #        pred_idx = pt['Pred.Token']
    #        first_arg = pt['arg_indices'][0]
    #        last_arg = pt['arg_indices'][-1]
    #        if first_arg < pred_idx:
    #            assert last_arg < pred_idx
    #        elif first_arg > pred_idx:
    #            assert last_arg > pred_idx
    #        else: # Arg index NEVER equal pred index
    #            raise Exception



if __name__ == '__main__':
    unittest.main()
