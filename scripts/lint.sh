#!/usr/bin/env bash

set -ev

if [[ -z $GITHUB_ACTION ]]; then
  ruff format openg2g tests examples
else
  ruff format --check openg2g tests examples
fi

ruff check openg2g tests examples
pyright openg2g tests examples
