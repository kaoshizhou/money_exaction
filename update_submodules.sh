git submodule init # will be skipped automatically if initted
git submodule update --remote --merge
git submodule foreach "(git checkout main)"