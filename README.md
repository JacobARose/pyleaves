PyLeaves
========

Shared repository for code related to the fossils project in Thomas Serre's lab at Brown University.
Full project repo containing sub-packages for database, dataloaders, models, training, and analysis.


### INSTALLATION: 
**To install, follow procedure below:**

    1. Clone git repo:
        >> git clone https://github.com/JacobARose/pyleaves.git

    2. Navigate to pyleaves root directory:
        >> cd pyleaves

    3. Make sure to create/activate environment:
        -- Implement "ENVIRONMENT MANAGEMENT" section below
        
    4. Install pyleaves with pip, include -e option to make sure it's an editable installation:
        >> pip install -e .

### ENVIRONMENT MANAGEMENT:
This can be used as a base reference, but individual packages may need to be installed in a largely trial and error fashion atm. Creating a minimum environment specification is TBD.

    1. Locate up-to-date environment spec file 'pyleaves.yml', usually in root pyleaves directory

    2. Create conda environment from spec file:
        >> conda env create -f pyleaves.yml -n pyleaves
        
    3. Activate conda environment:
        >> conda activate pyleaves
    
    4. Install latest version of pyleaves:
        >> git pull

Note, where it says "-n pyleaves' above, you can replace 'pyleaves" with your preferred choice of env name.

#### (ENVIRONMENT EXTRAS)
Only relevant when making changes to repo that require additional dependencies

    5a. Export conda environment spec file (Only do this if making major changes to requirements or working on new git branch):
        >> conda env export > pyleaves.yml
        
    5b. Export conda environment spec file using env history rather than list of specific versions
        >> conda env export --from-history > pyleaves_from_history.yml
        
            -- Useful option for when working on new machine and experience compatibility issues


GETTING STARTED: Take a look at pyleaves_demo.ipynb in the root directory for an example of how to interact with the database and query data. For further reference, see documentation for SQLAlchemy.


### DATABASE MANAGEMENT:

The most up-to-date json master record of datasets should always be found in './pyleaves/leavesdb/resources' under the filename 'full_dataset_frozen.json'. This what's used to create the SQLite database that enables efficient data management on a per-experiment basis. 

The format is standardized to allow easy human readability, in addition to flexible modularity for adding new datasets or new metadata related to current entries.

When pulling from the latest repo, one should be sure to recreate the db from the corresponding master json. This can be done using the default args in create_db.py, located in the root directory of pyleaves.

(a) **Main Usage:** 
Run from the cmd line with defaults:

    >> python create_db.py

(b) **Custom Usage:**
Run from the cmd line with custom json location or db location (db file is auto-named leavesdb.db in create_db.py, its parent dir is customizeable):

    >> python create_db.py --json_path  'PATH/TO/SOURCE/FILE.json' --output_folder 'PATH/TO/RESOURCES/DIR'

### MODIFICATION:
To modify and push changes to git:

    1. Navigate to root directory of package (/pyleaves)

    2. git add [list of files to commit] or git add --all

    3. git commit -m "message describing changes"

    4. git push

    5. '[Enter GitHub login credentials]'
