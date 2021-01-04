import ruamel.yaml as ruamel_yaml
import io
import os
import collections
import json

class ConfigDict(dict):
    """
    Dictionary with access to all config files
    """

    def __init__(self, *args, **kwargs):
        if len(args) > 0:
            super().__init__(args[0])
        else:
            super().__init__(kwargs)

    @staticmethod
    def read(path):
        """
        read dictionary from file / files
        Args:
            path (str): can be path to yaml/yml file or path to folder with
                yaml/yml files
        Returns:
            AttributeDict: read in dictionary
        """
        # get files
        if os.path.isfile(path):
            fnames = [path]
        else:
            fnames = [os.path.join(path, fname) for fname in os.listdir(path)]
        fnames = [fname for fname in fnames if fname.endswith('.yaml') or fname.endswith('.yml')]
        # read
        config = ConfigDict()
        for fname in fnames:
            with open(fname, 'r', encoding='utf-8') as f:
                part_config = ruamel_yaml.YAML().load(f)
            config.update(ConfigDict.from_dict(part_config))
        return config


    @staticmethod
    def from_dict(dict_like):
        """
        recursively converting all dicts to AttributeDict - removes all comments from yaml
        Args:
            dict_like (dict): dict-like object
        Returns:
            AttributeDict
        """

        def convert(value):
            if isinstance(value, dict):
                return ConfigDict.from_dict(value)
            else:
                return value

        return ConfigDict(**{key: convert(value) for key, value in dict_like.items()})

    def __setattr__(self, key, value):
        self[key] = value

    def __setitem__(self, key, value):
        if isinstance(value, dict) and not isinstance(value, ConfigDict):
            value = ConfigDict(**value)
        super().__setitem__(key, value)

    def __dir__(self):
        return self.keys()

    def __hash__(self):
        # try hashing: will work for items loaded from yaml
        return hash(json.dumps(self.to_dict(), sort_keys=True))

    # for backwards compatibility
    @property
    def _d(self):
        return self

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)