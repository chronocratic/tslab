"""The launch backends: one object per launcher, SLURM and local apart.

A study command asks the backend of the machine it runs on, picked once by
``backend_for`` from the profile's launcher; it never names a launcher. The
backend code lives in the ``slurm`` and ``local`` subpackages, which never
import each other; what they share lives in ``common``, and ``tables`` is the
one module that names both.
"""
