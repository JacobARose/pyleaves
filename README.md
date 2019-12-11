# pyleaves
Shared repository for code related to the fossils project in Thomas Serre's lab at Brown University.
Full project repo containing sub-packages for database, dataloaders, models, training, and analysis.


INSTALLATION: To install, follow procedure below:

    1. git clone https://github.com/JacobARose/pyleaves.git

    2. cd pyleaves

    3. pip install -e .

ENVIRONMENT SETUP: Note, this is simply a cloning of the conda environment used to create the package, and thus is not a minimum set of requirements. Creating a minimum environment specification is TBD.

    1. Navigate to the root /pyleaves directory containing environment spec file 'pyleaves.yml'

    2. conda env create -f pyleaves.yml -n pyleaves

Note, where it says "-n pyleaves' above, you can replace 'pyleaves" with your preferred choice of env name.

GETTING STARTED: Take a look at pyleaves_demo.ipynb in the root directory for an example of how to interact with the database and query data.


Db management
=============

The folder resources contains the latest .db file for the paths, and the source json file for the structure. If any change is made to include other datasets, it should be updated in the repo. 

To create db, there is a hihglevel script: create_db.py. You can: 

'''python
    python create_db.py --json_path  'PATHTONEWJSON' --output_folder 'PATHTORESOURCES'
'''



MODIFICATION: To modify and push changes to git:

    1. Navigate to root directory of package (/pyleaves)

    2. git add [list of files to commit] or git add --all

    3. git commit -m "message describing changes"

    4. git push

    5. '[Enter GitHub login credentials]'
