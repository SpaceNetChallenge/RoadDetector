#!/usr/bin/env bash

docker images -q --filter "dangling=true" | xargs docker rmi
