#!/bin/sh
if [ $# -eq 0 ]; then
    exec crond -f -l 8
else
    exec certbot "$@"
fi
