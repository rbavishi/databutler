import os
import unittest

from databutler.utils import paths, pickleutils


class _CustomObjectClass:
    def __init__(self, a: int):
        self.a = a

    def test_method(self):
        return self.a


class PickleUtilsTests(unittest.TestCase):
    def test_smart_dump_and_load_1(self):
        pickle_path = os.path.join(paths.get_tmpdir_path(), "pickleutils_test_1.pkl")
        pickleutils.smart_dump(10, pickle_path)
        self.assertEqual(10, pickleutils.smart_load(pickle_path))
        os.unlink(pickle_path)

    def test_smart_dump_and_load_2(self):
        pickle_path = os.path.join(paths.get_tmpdir_path(), "pickleutils_test_2.pkl")
        pickleutils.smart_dump(_CustomObjectClass(10), pickle_path)
        self.assertEqual(10, pickleutils.smart_load(pickle_path).test_method())
        os.unlink(pickle_path)

    def test_pickled_collections_1(self):
        pickle_path = os.path.join(paths.get_tmpdir_path(), "pickleutils_pc_test_1.pkl")
        with pickleutils.PickledCollectionWriter(pickle_path) as writer:
            writer.append(10)
            writer.append(20)
            writer.append(30)

        with pickleutils.PickledCollectionReader(pickle_path) as reader:
            self.assertEqual(10, reader[0])
            self.assertEqual(20, reader[1])
            self.assertEqual(30, reader[2])

        pickleutils.delete_pickled_collection(pickle_path)

    def test_pickled_collections_2(self):
        pickle_path = os.path.join(paths.get_tmpdir_path(), "pickleutils_pc_test_2.pkl")
        with pickleutils.PickledCollectionWriter(pickle_path) as writer:
            writer.append(10)
            writer.append(20)
            writer.append(30)

        with pickleutils.PickledCollectionWriter(
            pickle_path, overwrite_existing=False
        ) as writer:
            writer.append(40)
            writer.append(50)
            writer.append(60)

        with pickleutils.PickledCollectionReader(pickle_path) as reader:
            self.assertEqual(6, len(reader))
            self.assertEqual(10, reader[0])
            self.assertEqual(20, reader[1])
            self.assertEqual(30, reader[2])
            self.assertEqual(40, reader[3])
            self.assertEqual(50, reader[4])
            self.assertEqual(60, reader[5])

        pickleutils.delete_pickled_collection(pickle_path)

    def test_pickled_map_1(self):
        pickle_path = os.path.join(paths.get_tmpdir_path(), "pickleutils_pm_test_1.pkl")
        with pickleutils.PickledMapWriter(pickle_path) as writer:
            writer["10"] = 10
            writer["20"] = 20
            writer["30"] = 30

        with pickleutils.PickledMapReader(pickle_path) as reader:
            self.assertEqual(10, reader["10"])
            self.assertEqual(20, reader["20"])
            self.assertEqual(30, reader["30"])

        pickleutils.delete_pickled_map(pickle_path)

    def test_pickled_map_2(self):
        pickle_path = os.path.join(paths.get_tmpdir_path(), "pickleutils_pm_test_2.pkl")
        with pickleutils.PickledMapWriter(pickle_path) as writer:
            writer["10"] = 10
            writer["20"] = 20
            writer["30"] = 30

        with pickleutils.PickledMapWriter(
            pickle_path, overwrite_existing=False
        ) as writer:
            writer["40"] = 40
            writer["50"] = 50
            writer["60"] = 60

        with pickleutils.PickledMapReader(pickle_path) as reader:
            self.assertEqual(6, len(reader))
            self.assertEqual(10, reader["10"])
            self.assertEqual(20, reader["20"])
            self.assertEqual(30, reader["30"])
            self.assertEqual(40, reader["40"])
            self.assertEqual(50, reader["50"])
            self.assertEqual(60, reader["60"])

        pickleutils.delete_pickled_map(pickle_path)

    def test_pickled_map_3(self):
        pickle_path = os.path.join(paths.get_tmpdir_path(), "pickleutils_pm_test_2.pkl")
        with pickleutils.PickledMapWriter(pickle_path) as writer:
            writer["10"] = 10
            writer["20"] = 20
            writer["30"] = 30
            self.assertRaises(ValueError, writer.__setitem__, "10", 10)

        with pickleutils.PickledMapWriter(
            pickle_path, overwrite_existing=False
        ) as writer:
            writer["40"] = 40
            writer["50"] = 50
            writer["60"] = 60
            self.assertRaises(ValueError, writer.__setitem__, "10", 10)
            self.assertRaises(ValueError, writer.__setitem__, "40", 40)

        pickleutils.delete_pickled_map(pickle_path)
