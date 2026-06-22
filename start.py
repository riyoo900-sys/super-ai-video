#!/usr/bin/env python3
"""Entrypoint: patch diffusers, then start RunPod handler."""
import bootstrap  # noqa: F401

import handler

if __name__ == "__main__":
    handler.main()
