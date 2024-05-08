#!/usr/bin/bash
set -xe

ls *.zip | xargs -I{} unzip {}
