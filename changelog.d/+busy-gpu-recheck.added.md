With `skip_busy_gpus`, a local launch checks a GPU again before starting a
run on it, so a GPU taken by someone else during the launch gets no new runs.
