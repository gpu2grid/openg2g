#!/usr/bin/env bash

set -ev

FIX_FLAG=""
if [[ "$1" == "--fix" ]]; then
  FIX_FLAG="--fix"
fi

if [[ -z $GITHUB_ACTION ]]; then
  ruff format openg2g tests examples
else
  ruff format --check openg2g tests examples
fi

ruff check $FIX_FLAG openg2g tests examples
ty check openg2g tests examples
