A local launch no longer holds the submission lock of the study while its tasks run; it releases the lock once it holds the pid file, and a second launch still fails on the pid file.
