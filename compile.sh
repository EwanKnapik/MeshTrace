#!/bin/bash

cd submodules/diff-triangle-mesh-rasterization/

rm -rf build
rm -rf dist
rm -rf diff_triangle_rasterization.egg-info

pip install . --no-build-isolation

cd ../..
