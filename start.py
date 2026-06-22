#!/usr/bin/env python3
"""Entrypoint — patch only, no heavy imports at startup."""
import bootstrap  # noqa: F401

if __name__ == "__main__":
    import handler

    handler.main()
