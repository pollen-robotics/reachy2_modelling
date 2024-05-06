#!/usr/bin/bash
set -xe

ls */*.csv | xargs -I{} cp -v {} {}.bk
