# adaptive-rlvr

The statistics of adaptive rollout collection in RL with verifiable rewards.

`theory/exact.py` checks, by exact enumeration, that for binary rewards every collection
scheme whose sampling, stopping, filtering, and advantages depend only on a prompt's
rewards has (1) an expected update equal to a scalar c_S(p) times the gradient of the
prompt's success probability, with c_S depending only on p, and (2) a second moment fixed
by four numbers per prompt. Dynamic sampling (dropping constant groups) only rescales the
batch update. Token-mean length normalization and non-binary rewards break the structure.

```
cd theory && python3 exact.py
```
