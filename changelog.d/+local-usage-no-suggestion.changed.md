`tslab study usage` on a local machine no longer suggests a time limit: the `suggested_time` of each group is `null`, and the `local: suggest timeout=` line is gone.
On any machine, `tslab study usage` no longer logs a `suggest time=None mem_per_cpu=None` line for a group that has no suggestion.
