import numpy as np
import scipy.io as sio


import os
import wget
from zipfile import ZipFile 

class DataReader:

    def __init__(self, data='Movementdata'):
 
        self.downloadandunzip(data)
        self.path_to_data = os.path.join('data', data)
        self.path_to_train = os.path.join(self.path_to_data,'train')
        self.path_to_valid = os.path.join(self.path_to_data,'validation')

        #print(self.path_to_train)
        #print(self.path_to_valid)

        self.train_data, self.valid_data = self.read_data()   


    def read_data(self):
        
        train_data_names = os.listdir(self.path_to_train)
        valid_data_names = os.listdir(self.path_to_valid)

        train_data = []
        for t in train_data_names:
            t_p = os.path.join(self.path_to_train, t)
            train_data.append(t_p)
        train_data_names = train_data

        valid_data = []
        for t in valid_data_names:
            t_p = os.path.join(self.path_to_valid, t)
            valid_data.append(t_p)
        valid_data_names = valid_data


        train_data = []
        for name in train_data_names:
            d = sio.loadmat(name)['BD']
            train_data.append(d)
        print(f'There is total of {len(train_data)} training data')

        valid_data = []
        for name in valid_data_names:
            d = sio.loadmat(name)['BD']
            valid_data.append(d)
        print(f'There is total of {len(valid_data)} Validation data')

        return train_data, valid_data

    def get_data(self):
        return self.train_data, self.valid_data

    @staticmethod
    def downloadandunzip(data='Movementdata'):
        
        pathtodata = os.path.join('data', data)

        # Check if Data Folder exist
        if os.path.exists(pathtodata):
            print('Folder exists')
            
            # Check if train data exists
            trainPath = os.path.join(pathtodata, 'train.zip')
            validPath = os.path.join(pathtodata, 'validation.zip')
            
            if os.path.exists(trainPath):
                print('Training zip file exists')
            else:
                print('No Training Data')
                print('Now downloading training data')
                wget.download(url='https://comp0053-emopain.s3.eu-west-2.amazonaws.com/movementData/train.zip',
                            out=pathtodata)

            if os.path.exists(validPath):
                print('Validation zip file exists')
            else:
                print('Now downloading Validation data')
                wget.download(url='https://comp0053-emopain.s3.eu-west-2.amazonaws.com/movementData/validation.zip',
                            out=pathtodata)

            # Unzip
            assert os.path.exists(trainPath)
            assert os.path.exists(validPath)

            # Create
            train_dest = os.path.join(pathtodata, 'train')
            valid_dest = os.path.join(pathtodata, 'validation')
            

            if os.path.exists(train_dest):
                print('Train data already exists')
            else:
                #os.mkdir(train_dest)
                with ZipFile(trainPath, 'r') as zipObj:
                    zipObj.extractall(pathtodata)


            if os.path.exists(valid_dest):
                print('Valid data already exists')
            else:
                #os.mkdir(valid_dest)
                with ZipFile(validPath, 'r') as zipObj:
                    zipObj.extractall(pathtodata)   

        else:
            # Create a data folder
            print(f'Create a ->{data}<- folder ')
            os.mkdir(pathtodata)
            
            print('\nNow downloading training data')
            wget.download(url='https://comp0053-emopain.s3.eu-west-2.amazonaws.com/movementData/train.zip',
                        out=pathtodata)
            
            print('\nNow downloading Validation data')
            wget.download(url='https://comp0053-emopain.s3.eu-west-2.amazonaws.com/movementData/validation.zip',
                        out=pathtodata)

            trainPath = os.path.join(pathtodata, 'train.zip')
            validPath = os.path.join(pathtodata, 'validation.zip')

            print('Unzipping training data')
            with ZipFile(trainPath, 'r') as zipObj:
                zipObj.extractall(pathtodata)

            print('Unzipping validation data')
            with ZipFile(validPath, 'r') as zipObj:
                zipObj.extractall(pathtodata)   

'''
def downloadandunzip(data='Movementdata'):
    
    pathtodata = os.path.join('data', data)

    # Check if Data Folder exist
    if os.path.exists(pathtodata):
        print('Folder exists')
        
        # Check if train data exists
        trainPath = os.path.join(pathtodata, 'train.zip')
        validPath = os.path.join(pathtodata, 'validation.zip')
        
        if os.path.exists(trainPath):
            print('Training zip file exists')
        else:
            print('No Training Data')
            print('Now downloading training data')
            wget.download(url='https://comp0053-emopain.s3.eu-west-2.amazonaws.com/movementData/train.zip',
                        out=pathtodata)

        if os.path.exists(validPath):
            print('Validation zip file exists')
        else:
            print('Now downloading Validation data')
            wget.download(url='https://comp0053-emopain.s3.eu-west-2.amazonaws.com/movementData/validation.zip',
                        out=pathtodata)

        # Unzip
        assert os.path.exists(trainPath)
        assert os.path.exists(validPath)

        # Create
        train_dest = os.path.join(pathtodata, 'train')
        valid_dest = os.path.join(pathtodata, 'validation')
        

        if os.path.exists(train_dest):
            print('Train data already exists')
        else:
            #os.mkdir(train_dest)
            with ZipFile(trainPath, 'r') as zipObj:
                zipObj.extractall(pathtodata)


        if os.path.exists(valid_dest):
            print('Valid data already exists')
        else:
            #os.mkdir(valid_dest)
            with ZipFile(validPath, 'r') as zipObj:
                zipObj.extractall(pathtodata)   

    else:
        # Create a data folder
        print(f'Create a ->{data}<- folder ')
        os.mkdir(pathtodata)
        
        print('\nNow downloading training data')
        wget.download(url='https://comp0053-emopain.s3.eu-west-2.amazonaws.com/movementData/train.zip',
                    out=pathtodata)
        
        print('\nNow downloading Validation data')
        wget.download(url='https://comp0053-emopain.s3.eu-west-2.amazonaws.com/movementData/validation.zip',
                    out=pathtodata)

        trainPath = os.path.join(pathtodata, 'train.zip')
        validPath = os.path.join(pathtodata, 'validation.zip')

        print('Unzipping training data')
        with ZipFile(trainPath, 'r') as zipObj:
            zipObj.extractall(pathtodata)

        print('Unzipping validation data')
        with ZipFile(validPath, 'r') as zipObj:
            zipObj.extractall(pathtodata)   
'''

if __name__ == "__main__":
    pass
    #Test code
    '''
    downloadandunzip()
    dl= DataReader()
    a, b = dl.get_data()
    print(len(a), len(b))
    '''