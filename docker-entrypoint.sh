#!/bin/sh
set -e

# If the first argument starts with '-' or is a known server flag,
# prepend the python server command.
if [ "${1#-}" != "$1" ] || [ "$1" = "--sd-activate" ]; then
    set -- python server.py "$@"
fi

exec "$@"
