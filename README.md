# pyleaves
Shared repository for code related to the fossils project in Thomas Serre's lab at Brown University.
Full project repo containing sub-packages for database, dataloaders, models, training, and analysis.


## INSTALLATION: 
### To install, follow procedure below:

    1. git clone https://github.com/JacobARose/pyleaves.git

    2. cd pyleaves

    3. pip install -e .

## ENVIRONMENT SETUP:
### Note, this is simply a cloning of the conda environment used to create the package, and thus is not a minimum set of requirements. Creating a minimum environment specification is TBD.

    1. Navigate to the root /pyleaves directory containing environment spec file 'pyleaves.yml'

    2. conda env create -f pyleaves.yml -n pyleaves

Note, where it says "-n pyleaves' above, you can replace 'pyleaves" with your preferred choice of env name.

GETTING STARTED: Take a look at pyleaves_demo.ipynb in the root directory for an example of how to interact with the database and query data.


## DATABASE MANAGEMENT:
## =============

The most up-to-date json record of datasets should always be found in './pyleaves/leavesdb/resources' under the filename 'full_dataset_frozen.json'. This what's used to create the SQLite database that enables efficient data management on a per-experiment basis. 

The format is standardized to allow easy human readability, in addition to flexible modularity for adding new datasets or new metadata related to current entries.

When pulling from the latest repo, one should be sure to recreate the db from the corresponding json. This can be done using the default args in create_db.py, located in the root directory.

### *Run from the cmd line with defaults:*

    >> python create_db.py

### Run from the cmd line with custom json location or db location (db file is auto-named leavesdb.db in create_db.py, its parent dir is customizeable):

    >> python create_db.py --json_path  'PATH/TO/SOURCE/FILE.json' --output_folder 'PATH/TO/RESOURCES/DIR'

## MODIFICATION: To modify and push changes to git:

    1. Navigate to root directory of package (/pyleaves)

    2. git add [list of files to commit] or git add --all

    3. git commit -m "message describing changes"

    4. git push

    5. '[Enter GitHub login credentials]'
