The chain exit check of a task walks the run folder once, outside the submission lock, and asks the scheduler for the requeue limit only when a retry is possible.
