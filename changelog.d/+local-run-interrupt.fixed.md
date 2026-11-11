An interrupt or error while a local task runs now stops its process group, so no child keeps the GPU lease files open; `process_alive` is false for a pid of zero or below.
