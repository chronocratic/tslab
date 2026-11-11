A failing GPU lease check or wait callback in the local pool returns its slot and starts no new command, so the error shows once the running commands end; running commands are not stopped.
