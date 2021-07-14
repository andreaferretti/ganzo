# Copyright 2020 UniCredit S.p.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import importlib
from collections import defaultdict


class RegistryError(ValueError):
    pass

class Registry:
    _available = defaultdict(dict)
    _defaults = dict()
    _option_functions = []

    @staticmethod
    def add(namespace, name, cls):
        Registry._available[namespace][name] = cls

    @staticmethod
    def keys(namespace):
        return list(Registry._available[namespace].keys())

    @staticmethod
    def get(namespace, name):
        return Registry._available[namespace].get(name)

    @staticmethod
    def set_default(namespace, name):
        if namespace in Registry._defaults:
            raise RegistryError(f'namespace {namespace} already has a default: {Registry._defaults[namespace]}')
        Registry._defaults[namespace] = name

    @staticmethod
    def default(namespace):
        return Registry._defaults.get(namespace)

    @staticmethod
    def add_option_function(f):
        Registry._option_functions.append(f)

    @staticmethod
    def option_functions():
        return list(Registry._option_functions)

def register(namespace, name, default=False):
    def inner(cls):
        Registry.add(namespace, name, cls)
        if default:
            Registry.set_default(namespace, name)
        return cls

    return inner

def with_option_parser(f):
    Registry.add_option_function(f)


_external_modules_name = os.environ.get('GANZO_LOAD_MODULES')
if _external_modules_name is not None:
    for module_name in _external_modules_name.split(','):
        importlib.import_module(module_name)