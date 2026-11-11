`chain_exit` asks the scheduler for `MaxBatchRequeue` before it takes the submission lock, skips the folder grace wait under the lock, and reports only a busy lock as a busy lock.
