"""Read segment IDs without reconstructing numeric feature or label arrays.

The trusted local attachment stores numeric tensors as numpy ndarray pickle
states. Replace those constructors with metadata stubs; return only train/valid
string IDs. No numeric array is materialized or any test label returned.
"""
import pickle
import re


class ArrayMetadata:
    def __setstate__(self, state):
        self.shape = state[1]
        self.strings = state[4] if state[2].kind == 'O' else None


class MetadataUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == '_reconstruct' and 'numpy' in module:
            return lambda *args: ArrayMetadata()
        if name == 'scalar' and 'numpy' in module:
            return lambda *args: None
        return super().find_class(module, name)


def read_group_ids(path):
    with open(path, 'rb') as handle:
        source = MetadataUnpickler(handle).load()
    output = {}
    for split in ('train', 'valid'):
        ids = source[split]['id']
        if isinstance(ids, ArrayMetadata):
            ids = ids.strings
        if not isinstance(ids, list) or not all(isinstance(v, str) and re.fullmatch(r'.+\$_\$\d+', v) for v in ids):
            raise ValueError('Unrecognized segment ID convention')
        if len(set(ids)) != len(ids):
            raise ValueError('Duplicate source segment IDs')
        # Metadata audit must not accidentally deserialize actual label ndarrays.
        for key in ('classification_labels', 'regression_labels'):
            if not isinstance(source[split][key], ArrayMetadata):
                raise ValueError('Unexpected label serialization; numeric isolation not verified')
        output[split] = dict(segment_ids=ids, group_ids=[v.rsplit('$_$', 1)[0] for v in ids])
    return output
