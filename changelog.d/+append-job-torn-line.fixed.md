Appending a job record now writes every byte and starts a new line after a torn record, so a record never glues onto a partial line; the in-place requeue holds the submission lock.
