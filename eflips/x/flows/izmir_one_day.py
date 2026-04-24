#!/usr/bin/env python3

"""
Izmir (IZBB) one-day flow — thin stub.

Runs the full Eshot network but only for a single day. See ``izmir.py`` for
the actual implementation; this file just selects the ``one_day`` variant.
"""
from eflips.x.flows.izmir import main

if __name__ == "__main__":
    main(variant="one_day")
