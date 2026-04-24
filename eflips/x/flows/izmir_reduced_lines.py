#!/usr/bin/env python3

"""
Izmir (IZBB) reduced-lines flow — thin stub.

Runs a subset of 12 Eshot routes for one week. See ``izmir.py`` for the
actual implementation; this file just selects the ``reduced_lines`` variant.
"""
from eflips.x.flows.izmir import main

if __name__ == "__main__":
    main(variant="reduced_lines")
