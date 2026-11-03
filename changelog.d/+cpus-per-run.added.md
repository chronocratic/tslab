New `{cpus_per_run}` placeholder: the CPUs of one run when several run at
once; local runs also get `OMP_NUM_THREADS` set to it unless it is already set.
