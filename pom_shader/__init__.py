# SPDX-License-Identifier: GPL-3.0-or-later

_MODULES = (
    "preferences",
    "props",
    "ops",
    "ui",
)


def _modules():
    import importlib

    return [importlib.import_module(f".{name}", __package__) for name in _MODULES]


def register() -> None:
    for mod in _modules():
        mod.register()


def unregister() -> None:
    for mod in reversed(_modules()):
        mod.unregister()
