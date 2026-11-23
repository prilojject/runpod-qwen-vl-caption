# Runpod Hub always builds from a Dockerfile. This thin file lets the Hub
# reuse the already-built public image instead of compiling the worker again.
# The real build recipe lives in the repository-root Dockerfile.
#
# Pin the digest (not a floating :latest). Releases of this repo must still
# use targetCommitish=main — this digest is the image pin, not the git target.
FROM ghcr.io/prilojject/runpod-qwen-vl-caption@sha256:db066f89f6a3b7c76816458162079d640d5d8afe6bc3fbdff2d3520cda691869
