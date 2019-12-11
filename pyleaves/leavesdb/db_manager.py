import pandas as pd 
import dataset
from .db_utils import load, flattenit, image_checker
import cv2 
import os

PATH='/media/data_cifs/irodri15/data/processed/datasets.json'
OUTPUT = 'sqlite:///resources/leavesdb.db'

def create_db(jsonpath=PATH,folder= 'Resources'):
    '''
    File to create a db from a json file. check the structure of the Json.
    The function would look for the key 'paths' as a stop key. 
    Arguments:
        - Json file with the following structure: 
            file: { 'dataset1' : {'family1': 
                                        'genus1':{ 
                                            'specie1':{
                                                'paths': [path1.jpg,...],
                                                .....
                                            },
                                            ...    
                                        },
                                  
                                  'family2':{},.. , ...}}
    Returns
    '''
    file = load(jsonpath)
    db_path = os.path.join(folder,'leavesdb.db')
    print(db_path)
    output = f'sqlite:///{db_path}'
    print(output)
    db = dataset.connect(output)
    db.begin()

    table = db['dataset']
    counter= 0 
    invalid_images=[]
    for data_set in file:
        res = {k:v for k, v in flattenit(file[data_set])}
        print(data_set)
        for key in res: 
            if 'paths' in key:
                names = key.split('_')[:-1]
                if len(names)==1:
                    continue 
                    print(names[0])
                    for p  in res[key]:
                        sample= dict(path=p,
                                      dataset=data_set,
                                      family=names[0],
                                      specie='nn',
                                      genus='nn'
                                      )
                        if image_checker:
                            table.insert(sample)
                            counter+=1
                        else: 
                            invalid_images.append([p,data_set,family,'nn','nn'])
                            

                else:   
                    print(names)
                    for p  in res[key]:
                        family = names[0]
                        if 'uncertain' in family:
                            family='uncertain'
                        if image_checker(p):
                            
                            sample= dict(path=p,
                                      dataset=data_set,
                                      family=names[0],
                                      specie=names[2],
                                      genus=names[1]
                                      )
                            table.insert(sample)
                            counter+=1
                        else: 
                            invalid_images.append([p,data_set,names[0],names[2],names[1]])
                        if counter%1000==0:
                            print(counter)
                            
    df_inv = pd.DataFrame(invalid_images,columns=['path','dataset','family','specie','genus'])
    output_csv= os.path.join(folder,'invalid_paths.csv')
    df_inv.to_csv(output_csv)
    db.commit()

#def onboard_dataset(jsonpath='newdataset.json', currentdb='sqlite:///leavesdb.db'):
#
#    file =load(jsonpath)
#    db = dataset.connect(currentdb)
#    with db as tb1:
#        try:
#            tb1['dataset'].insert()