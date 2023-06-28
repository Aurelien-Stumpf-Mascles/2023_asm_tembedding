To build container:
- rsync -av ../../2023_asm_tembedding /tmp
- cd /tmp/2023_asm_tembedding/singularity
- sudo singularity build tembedding.sif tembedding.def

To run the container with to precomute the distances between dFCs:
- singularity run tembedding.sif precompute-dfc-dist --help
- singularity run --bind <folders> tembedding.sif precompute-dfc-dist <command line parameters>
