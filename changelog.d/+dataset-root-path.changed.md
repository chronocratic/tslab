`ucr_datasets` and `uea_datasets` take `root` as a `pathlib.Path`; a string raises `TypeError`. `InstanceRecord.traceback_file` is a `Path`; `status.json` keeps the same text.
