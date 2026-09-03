# TODO
- Update passes
    - sequence - rename so it also supports parallel subgraphs detection.
    - merge the new 'sequence' with 'batch'
- Verify batching pass

Writing the paper... :
- the implementation still exposes 7 passes instead of 5
- MLIR is outdated, liekly not compiling anymore
- prog-fuse not always generates a PACKED version; it should.
- in CGIR implementation, there is actually more function arguments convention, but they could be simplified to the one presented in the paper
