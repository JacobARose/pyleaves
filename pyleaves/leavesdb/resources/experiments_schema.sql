/**
 * @Author: Jacob A Rose
 * @Date:   Wed, April 15th 2020, 3:44 am
 * @Email:  jacobrose@brown.edu
 * @Filename: experiments_schema.sql
 * @Last modified by:   Jacob A Rose
 * @Last modified time: Fri, April 17th 2020, 5:49 pm
 */



 PRAGMA foreign_keys = ON;

 DROP TABLE IF EXISTS experiments;

 CREATE TABLE experiments (
 	experiment_type TEXT PRIMARY KEY,
 	experiment_description TEXT,
 	num_runs INTEGER
 );

 DROP TABLE IF EXISTS runs;

 CREATE TABLE runs (
 	run_id    TEXT PRIMARY KEY,
 	experiment_type TEXT NOT NULL,
 	dataset_A TEXT NOT NULL,
 	dataset_B TEXT,
 	FOREIGN KEY (experiment_type)	REFERENCES experiments (experiment_type)
        ON UPDATE CASCADE
        ON DELETE CASCADE
 );

 DROP TABLE IF EXISTS subruns;

 CREATE TABLE subruns (
    subrun_id       TEXT PRIMARY KEY,
 	run_id          TEXT NOT NULL,
 	experiment_type TEXT NOT NULL,
 	dataset_A       TEXT NOT NULL,
 	dataset_B       TEXT,
    dataset_test    TEXT,

    FOREIGN KEY (run_id) REFERENCES runs (run_id)
       ON UPDATE CASCADE
       ON DELETE CASCADE,
 	FOREIGN KEY (experiment_type)	REFERENCES experiments (experiment_type)
        ON UPDATE CASCADE
        ON DELETE CASCADE
 );


DROP TABLE IF EXISTS parameters;

CREATE TABLE parameters (
    subrun_id       TEXT PRIMARY KEY,
	run_id          TEXT NOT NULL,
	experiment_type TEXT NOT NULL,


    FOREIGN KEY (run_id) REFERENCES runs (run_id)
        ON UPDATE CASCADE
        ON DELETE CASCADE,
	FOREIGN KEY (experiment_type)	REFERENCES experiments (experiment_type)
        ON UPDATE CASCADE
        ON DELETE CASCADE
);





 DROP TABLE IF EXISTS tfrecords;

 CREATE TABLE tfrecords (
	file_path	    TEXT PRIMARY KEY,
    file_group	    TEXT NOT NULL,
    dataset_stage   TEXT default "dataset_A",
    subrun_id       TEXT,
	run_id    	    TEXT NOT NULL,
	experiment_type TEXT NOT NULL,
	dataset_name    TEXT NOT NULL,
    resolution      INTEGER NOT NULL,
    num_channels    INTEGER NOT NULL,
    num_classes     INTEGER NOT NULL,
    num_shards      INTEGER NOT NULL,
    num_samples     INTEGER NOT NULL,
    FOREIGN KEY (subrun_id) REFERENCES subruns (subrun_id)
       ON UPDATE CASCADE
       ON DELETE CASCADE,
	FOREIGN KEY (run_id) REFERENCES runs (run_id)
       ON UPDATE CASCADE
       ON DELETE CASCADE,
	FOREIGN KEY (experiment_type) REFERENCES experiments (experiment_type)
       ON UPDATE CASCADE
       ON DELETE CASCADE
);


 INSERT INTO experiments
 VALUES
    ("A_train_val_test","single-dataset_40-10-50 train-val-test","3");

 INSERT INTO runs
 VALUES
    ("1000","A_train_val_test","PNAS", NULL);
 INSERT INTO runs
 VALUES
    ("1100","A_train_val_test","Fossil", NULL);
 INSERT INTO runs
 VALUES
    ("1200","A_train_val_test","Leaves", NULL);


 INSERT INTO experiments
 VALUES
    ("A+B_train_val_test","double-dataset_40-10-50 train-val-test","3");

 INSERT INTO runs
 VALUES
    ("2000","A+B_train_val_test","PNAS","Fossil");
 INSERT INTO runs
 VALUES
    ("2100","A+B_train_val_test","PNAS","Leaves");
 INSERT INTO runs
 VALUES
    ("2200","A+B_train_val_test","Fossil","Leaves");

 INSERT INTO experiments
 VALUES
    ("A_train_val-B_train_val_test","source2target-domain-transfer_50-50-train-test_40-10-50_train-val-test","3");

 INSERT INTO runs
 VALUES
    ("3000","A_train_val-B_train_val_test","PNAS", "Fossil");
 INSERT INTO runs
 VALUES
    ("3100","A_train_val-B_train_val_test","PNAS", "Leaves");
 INSERT INTO runs
 VALUES
    ("3200","A_train_val-B_train_val_test","Fossil", "Leaves");


 INSERT INTO experiments
 VALUES
    ("A+B_leave_one_out","LeaveOneOut_double-dataset_50-50_train-val_LeftOutClass_test","3");

 INSERT INTO runs
 VALUES
    ("4000","A+B_leave_one_out","PNAS","Fossil");
 INSERT INTO runs
 VALUES
    ("4100","A+B_leave_one_out","Leaves","Fossil");
 INSERT INTO runs
 VALUES
    ("4200","A+B_leave_one_out","Fossil", "Leaves");


INSERT INTO subruns
VALUES
   ("a","4000","A+B_leave_one_out","PNAS","Fossil","test1");
INSERT INTO subruns
VALUES
   ("b","4100","A+B_leave_one_out","Leaves","Fossil","test2");
INSERT INTO subruns
VALUES
   ("c","4200","A+B_leave_one_out","Fossil", "Leaves","test3");
