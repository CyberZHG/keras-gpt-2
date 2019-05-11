#!/usr/bin/env bash
pycodestyle --max-line-length=120 keras_gpt_2 tests && \
    nosetests --with-coverage --cover-erase --cover-html --cover-html-dir=htmlcov --cover-package=keras_gpt_2 tests
