#!/usr/bin/env bash


# Open workspace in current window when code is available on container
until code -r workspaces.code-workspace
do
    echo "Waiting for code to be installed on container."
    sleep 2
done