import os
import unittest

from databutler.utils import lazyobjs, pickleutils, paths


class _CustomObjectClass:
    def __init__(self, a: int):
        self.a = a

    def test_method(self):
        return self.a


class LazyObjTests(unittest.TestCase):
    def test_obj_ref_1(self):
        pickle_path = os.path.join(paths.get_tmpdir_path(), "lazyobjs_test_1.pkl")
        pickleutils.smart_dump(10, pickle_path)

        ref = pickleutils.PickledRef(pickle_path)
        self.assertEqual(10, ref.resolve())
        os.unlink(pickle_path)

    def test_obj_ref_2(self):
        pickle_path = os.path.join(paths.get_tmpdir_path(), "lazyobjs_test_2.pkl")
        pickleutils.smart_dump(_CustomObjectClass(10), pickle_path)

        ref = pickleutils.PickledRef(pickle_path)
        self.assertEqual(10, ref.resolve().test_method())
        os.unlink(pickle_path)

    def test_lazy_list_1(self):
        pickle_path = os.path.join(paths.get_tmpdir_path(), "lazyobjs_test_list_1.pkl")
        with pickleutils.PickledCollectionWriter(pickle_path) as writer:
            writer.append(10)
            writer.append(20)
            writer.append(30)

        ref1 = pickleutils.PickledRef(path=pickle_path, index=0)
        ref2 = pickleutils.PickledRef(path=pickle_path, index=1)
        ref3 = pickleutils.PickledRef(path=pickle_path, index=2)

        lazy_list = lazyobjs.LazyList([ref1, ref2, ref3])
        self.assertListEqual([10, 20, 30], lazy_list.resolve())

        os.unlink(pickle_path)

    def test_lazy_tuple_1(self):
        pickle_path = os.path.join(paths.get_tmpdir_path(), "lazyobjs_test_tuple_1.pkl")
        with pickleutils.PickledCollectionWriter(pickle_path) as writer:
            writer.append(10)
            writer.append(20)
            writer.append(30)

        ref1 = pickleutils.PickledRef(path=pickle_path, index=0)
        ref2 = pickleutils.PickledRef(path=pickle_path, index=1)
        ref3 = pickleutils.PickledRef(path=pickle_path, index=2)

        lazy_tuple = lazyobjs.LazyTuple((ref1, ref2, ref3))
        self.assertTupleEqual((10, 20, 30), lazy_tuple.resolve())

        os.unlink(pickle_path)

    def test_lazy_set_1(self):
        pickle_path = os.path.join(paths.get_tmpdir_path(), "lazyobjs_test_set_1.pkl")
        with pickleutils.PickledCollectionWriter(pickle_path) as writer:
            writer.append(10)
            writer.append(20)
            writer.append(30)

        ref1 = pickleutils.PickledRef(path=pickle_path, index=0)
        ref2 = pickleutils.PickledRef(path=pickle_path, index=1)
        ref3 = pickleutils.PickledRef(path=pickle_path, index=2)

        lazy_set = lazyobjs.LazySet((ref1, ref2, ref3))
        self.assertSetEqual({10, 20, 30}, lazy_set.resolve())

        os.unlink(pickle_path)

    def test_lazy_dict_1(self):
        pickle_path = os.path.join(paths.get_tmpdir_path(), "lazyobjs_test_dict_1.pkl")
        with pickleutils.PickledCollectionWriter(pickle_path) as writer:
            writer.append(10)
            writer.append(20)
            writer.append(30)

        ref1 = pickleutils.PickledRef(path=pickle_path, index=0)
        ref2 = pickleutils.PickledRef(path=pickle_path, index=1)
        ref3 = pickleutils.PickledRef(path=pickle_path, index=2)

        lazy_dict = lazyobjs.LazyDict(
            {
                1: ref1,
                2: ref2,
                3: ref3,
            }
        )
        self.assertDictEqual({1: 10, 2: 20, 3: 30}, lazy_dict.resolve())

        os.unlink(pickle_path)
