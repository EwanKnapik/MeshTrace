#!/bin/bash

cd submodules/diff-triangle-mesh-rasterization/

rm -rf build
rm -rf dist
rm -rf diff_triangle_rasterization.egg-info

pip install . --no-build-isolation

cd ../diff-triangle-feature-rasterization/

rm -rf build
rm -rf dist
rm -rf diff_triangle_feature_rasterization.egg-info

pip install . --no-build-isolation

cd ../..
