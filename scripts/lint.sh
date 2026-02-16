#!/usr/bin/env bash

set -ev

if [[ -z $GITHUB_ACTION ]]; then
  ruff format openg2g tests examples data
else
  ruff format --check openg2g tests examples data
fi

ruff check openg2g tests examples data
pyright openg2g tests examples data
